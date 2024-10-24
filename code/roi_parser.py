import os
import struct
import zipfile
import logging
import numpy as np
import cv2
import re
import ntpath
import argparse
from typing import Dict, List, Tuple, Union, Optional


class UnrecognizedRoiType(Exception):
    pass


OFFSET = dict(VERSION_OFFSET=4,
              TYPE=6,
              TOP=8,
              LEFT=10,
              BOTTOM=12,
              RIGHT=14,
              N_COORDINATES=16,
              X1=18,
              Y1=22,
              X2=26,
              Y2=30,
              XD=18,
              YD=22,
              WIDTHD=26,
              HEIGHTD=30,
              STROKE_WIDTH=34,
              SHAPE_ROI_SIZE=36,
              STROKE_COLOR=40,
              FILL_COLOR=44,
              SUBTYPE=48,
              OPTIONS=50,
              ARROW_STYLE=52,
              ELLIPSE_ASPECT_RATIO=52,
              ARROW_HEAD_SIZE=53,
              ROUNDED_RECT_ARC_SIZE=54,
              POSITION=56,
              HEADER2_OFFSET=60,
              COORDINATES=64)

ROI_TYPE = dict(polygon=0,
                rect=1,
                oval=2,
                line=3,
                freeline=4,
                polyline=5,
                noRoi=6,
                freehand=7,
                traced=8,
                angle=9,
                point=10)

OPTIONS = dict(SPLINE_FIT=1,
               DOUBLE_HEADED=2,
               OUTLINE=4,
               OVERLAY_LABELS=8,
               OVERLAY_NAMES=16,
               OVERLAY_BACKGROUNDS=32,
               OVERLAY_BOLD=64,
               SUB_PIXEL_RESOLUTION=128,
               DRAW_OFFSET=256)

HEADER_OFFSET = dict(C_POSITION=4,
                     Z_POSITION=8,
                     T_POSITION=12,
                     NAME_OFFSET=16,
                     NAME_LENGTH=20,
                     OVERLAY_LABEL_COLOR=24,
                     OVERLAY_FONT_SIZE=28,
                     AVAILABLE_BYTE1=30,
                     IMAGE_OPACITY=31,
                     IMAGE_SIZE=32,
                     FLOAT_STROKE_WIDTH=36,
                     ROI_PROPS_OFFSET=40,
                     ROI_PROPS_LENGTH=44,
                     COUNTERS_OFFSET=48)

SUBTYPES = dict(TEXT=1,
                ARROW=2,
                ELLIPSE=3,
                IMAGE=4)

# https://docs.oracle.com/javase/6/docs/api/constant-values.html#java.awt.geom.PathIterator
PATHITERATOR_TYPES = dict(SEG_MOVETO=0,
                          SEG_LINETO=1,
                          SEG_QUADTO=2,
                          SEG_CUBICTO=3,
                          SEG_CLOSE=4)


def _get_byte(data: bytes, base: Union[int, List[int]]) -> Union[int, List[int]]:
    if isinstance(base, int):
        return data[base]
    elif isinstance(base, list):
        return [data[b] for b in base]


def _get_uint16(data: bytes, base: int) -> int:
    b0 = data[base]
    b1 = data[base + 1]
    n = (b0 << 8) + b1
    return n


def _get_int16(data: bytes, base: int) -> int:
    n = _get_uint16(data, base)
    if n >= 32768:  # 2**15
        n -= 65536  # 2**16
    return n


def _get_maybe_int16(data: bytes, base: int, thr: int = 65036) -> int:
    """
    Load data which might be int16 or uint16.
    """
    n = _get_uint16(data, base)
    if thr < 32768:
        raise ValueError(
            "Threshold for distinguishing between int16 and uint16 must be"
            " at least 2^15 = 32768, but {} was given.".format(thr)
        )
    if n >= thr:
        # Convert to uint16
        n -= 65536  # 2**16
    return n


def _get_uint32(data: bytes, base: int) -> int:
    b0 = data[base]
    b1 = data[base + 1]
    b2 = data[base + 2]
    b3 = data[base + 3]
    n = ((b0 << 24) + (b1 << 16) + (b2 << 8) + b3)
    return n


def _get_float(data: bytes, base: int) -> float:
    s = struct.pack('I', _get_uint32(data, base))
    return struct.unpack('f', s)[0]


def _get_counter(data: bytes, base: int) -> Tuple[int, int]:
    """
    See setCounters() / getCounters() methods in IJ source, ij/gui/PointRoi.java.
    """

    b0 = data[base]
    b1 = data[base + 1]
    b2 = data[base + 2]
    b3 = data[base + 3]

    counter = b3
    position = (b1 << 8) + b2

    return counter, position


def _get_point_counters(data: bytes, hdr2Offset: int, n_coordinates: int, size: int) -> Optional[Tuple[List[int], List[int]]]:
    if hdr2Offset == 0:
        return None

    offset = _get_uint32(data, hdr2Offset + HEADER_OFFSET['COUNTERS_OFFSET'])
    if offset == 0:
        return None

    if offset + n_coordinates * 4 > size:
        return None

    counters = []
    positions = []
    for i in range(0, n_coordinates):
        cnt, position = _get_counter(data, offset + i * 4)
        counters.append(cnt)
        positions.append(position)

    return counters, positions


def _pathiterator2paths(shape_array: List[float]) -> List[List[Tuple[float, ...]]]:
    """
    Converts a shape array in PathIterator notation to polygon (or curved)
    paths.
    Parameters
    ----------
    shape_array : list of floats
        paths encoded in `java.awt.geom.PathIterator` format. Each segment
        within the path begins with a header value,
            0 : Move operation
            1 : Line segment
            2 : Quadratic segment
            3 : Cubic segment
            4 : Terminate path
        followed by a number of values describing the path along the segment
        to the next node. In the case of a termination operation, the current
        path ends, whilst for a move operation a new path begins with a new
        node described whose co-ordinate is given by the next two value in
        `shape_array`.
    Returns
    -------
    paths : list of lists of tuples
        The `segements` output contains a list of path paths. Each path
        is a list of points along the path. In its simplest form, each tuple
        in the list has length two, corresponding to a nodes along a polygon
        shape. Tuples of length 4 or 6 correspond to quadratic and cubic paths
        (respectively) from the previous node.
        ImageJ ROIs are only known to output linear segments (even with
        ellipses with subpixel precision enabled), so it is expected that all
        segments along the path should be length two tuples containing only the
        co-ordinates of the next point on the polygonal path.
    Notes
    -----
    Based on the ShapeRoi constructor "from an array of variable length path
    segments" and `makeShapeFromArray`, as found in:
    https://imagej.nih.gov/ij/developer/source/ij/gui/ShapeRoi.java.html
    With further reference to its `PathIterator` dependency, as found in:
    https://docs.oracle.com/javase/6/docs/api/constant-values.html#java.awt.geom.PathIterator
    """
    paths = []
    path = None
    i = 0
    while i < len(shape_array):
        segmentType = shape_array[i]
        if segmentType == PATHITERATOR_TYPES["SEG_MOVETO"]:
            # Move to
            if path is not None:
                paths.append(path)
            # Start a new segment with a node at this point
            path = []
            nCoords = 2
        elif segmentType == PATHITERATOR_TYPES["SEG_LINETO"]:
            # Line to next point
            nCoords = 2
        elif segmentType == PATHITERATOR_TYPES["SEG_QUADTO"]:
            # Quadratic curve to next point
            nCoords = 4
        elif segmentType == PATHITERATOR_TYPES["SEG_CUBICTO"]:
            # Cubic curve to next point
            nCoords = 6
        elif segmentType == PATHITERATOR_TYPES["SEG_CLOSE"]:
            # Segment close
            paths.append(path)
            path = None
            i += 1
            continue
        if path is None:
            raise ValueError("A path must begin with a move operation.")
        path.append(tuple(shape_array[i + 1: i + 1 + nCoords]))
        i += 1 + nCoords
    return paths


def _extract_basic_roi_data(data: bytes) -> Tuple[Dict[str, Union[str, int, float, List[Tuple[float, ...]]]], Tuple[int, int, int, int, int, int, int, int, int, int]]:
    size = len(data)
    code = '>'

    magic = _get_byte(data, list(range(4)))
    magic = "".join([chr(c) for c in magic])

    # TODO: raise error if magic != 'Iout'
    version = _get_uint16(data, OFFSET['VERSION_OFFSET'])
    roi_type = _get_byte(data, OFFSET['TYPE'])
    subtype = _get_uint16(data, OFFSET['SUBTYPE'])

    # Note that top, bottom, left, and right may be signed integers
    top = _get_maybe_int16(data, OFFSET['TOP'])
    left = _get_maybe_int16(data, OFFSET['LEFT'])
    if top >= 0:
        bottom = _get_uint16(data, OFFSET['BOTTOM'])
    else:
        bottom = _get_maybe_int16(data, OFFSET['BOTTOM'])
    if left >= 0:
        right = _get_uint16(data, OFFSET['RIGHT'])
    else:
        right = _get_maybe_int16(data, OFFSET['RIGHT'])
    width = right - left
    height = bottom - top

    n_coordinates = _get_uint16(data, OFFSET['N_COORDINATES'])
    options = _get_uint16(data, OFFSET['OPTIONS'])
    position = _get_uint32(data, OFFSET['POSITION'])
    hdr2Offset = _get_uint32(data, OFFSET['HEADER2_OFFSET'])

    logging.debug("n_coordinates: {}".format(n_coordinates))
    logging.debug("position: {}".format(position))
    logging.debug("options: {}".format(options))

    sub_pixel_resolution = (options == OPTIONS['SUB_PIXEL_RESOLUTION']) and version >= 222
    draw_offset = sub_pixel_resolution and (options == OPTIONS['DRAW_OFFSET'])
    sub_pixel_rect = version >= 223 and sub_pixel_resolution and (
            roi_type == ROI_TYPE['rect'] or roi_type == ROI_TYPE['oval'])

    logging.debug("sub_pixel_resolution: {}".format(sub_pixel_resolution))
    logging.debug("draw_offset: {}".format(draw_offset))
    logging.debug("sub_pixel_rect: {}".format(sub_pixel_rect))

    # Untested
    if sub_pixel_rect:
        xd = _get_float(data, OFFSET['XD'])
        yd = _get_float(data, OFFSET['YD'])
        widthd = _get_float(data, OFFSET['WIDTHD'])
        heightd = _get_float(data, OFFSET['HEIGHTD'])
        logging.debug("Entering in sub_pixel_rect")

    # Untested
    if hdr2Offset > 0 and hdr2Offset + HEADER_OFFSET['IMAGE_SIZE'] + 4 <= size:
        channel = _get_uint32(data, hdr2Offset + HEADER_OFFSET['C_POSITION'])
        slice = _get_uint32(data, hdr2Offset + HEADER_OFFSET['Z_POSITION'])
        frame = _get_uint32(data, hdr2Offset + HEADER_OFFSET['T_POSITION'])
        overlayLabelColor = _get_uint32(data, hdr2Offset + HEADER_OFFSET['OVERLAY_LABEL_COLOR'])
        overlayFontSize = _get_uint16(data, hdr2Offset + HEADER_OFFSET['OVERLAY_FONT_SIZE'])
        imageOpacity = _get_byte(data, hdr2Offset + HEADER_OFFSET['IMAGE_OPACITY'])
        imageSize = _get_uint32(data, hdr2Offset + HEADER_OFFSET['IMAGE_SIZE'])
        logging.debug("Entering in hdr2Offset")

    roi_props = (hdr2Offset, n_coordinates, roi_type, channel, slice, frame, position, version, subtype, size)

    shape_roi_size = _get_uint32(data, OFFSET['SHAPE_ROI_SIZE'])
    is_composite = shape_roi_size > 0

    if is_composite:
        roi = {'type': 'composite'}

        # Add bounding box rectangle details
        if sub_pixel_rect:
            roi.update(dict(left=xd, top=yd, width=widthd, height=heightd))
        else:
            roi.update(dict(left=left, top=top, width=width, height=height))

        # Load path iterator shape array and decode it into paths
        base = OFFSET['COORDINATES']
        shape_array = [_get_float(data, base + i * 4) for i in range(shape_roi_size)]
        roi['paths'] = _pathiterator2paths(shape_array)

        # NB: Handling position of roi is implemented in read_roi_file

        if version >= 218:
            # Not implemented
            # Read stroke width, stroke color and fill color
            pass
        if version >= 224:
            # Not implemented
            # Get ROI properties
            pass

        return roi, roi_props

    if roi_type == ROI_TYPE['rect']:
        roi = {'type': 'rectangle'}

        if sub_pixel_rect:
            roi.update(dict(left=xd, top=yd, width=widthd, height=heightd))
        else:
            roi.update(dict(left=left, top=top, width=width, height=height))

        roi['arc_size'] = _get_uint16(data, OFFSET['ROUNDED_RECT_ARC_SIZE'])

        return roi, roi_props

    elif roi_type == ROI_TYPE['oval']:
        roi = {'type': 'oval'}

        if sub_pixel_rect:
            roi.update(dict(left=xd, top=yd, width=widthd, height=heightd))
        else:
            roi.update(dict(left=left, top=top, width=width, height=height))

        return roi, roi_props

    elif roi_type == ROI_TYPE['line']:
        roi = {'type': 'line'}

        x1 = _get_float(data, OFFSET['X1'])
        y1 = _get_float(data, OFFSET['Y1'])
        x2 = _get_float(data, OFFSET['X2'])
        y2 = _get_float(data, OFFSET['Y2'])

        if subtype == SUBTYPES['ARROW']:
            # Not implemented
            pass
        else:
            roi.update(dict(x1=x1, x2=x2, y1=y1, y2=y2))
            roi['draw_offset'] = draw_offset

        strokeWidth = _get_uint16(data, OFFSET['STROKE_WIDTH'])
        roi.update(dict(width=strokeWidth))

        return roi, roi_props

    elif roi_type in [ROI_TYPE[t] for t in
                      ["polygon", "freehand", "traced", "polyline", "freeline", "angle", "point"]]:
        x = []
        y = []

        if sub_pixel_resolution:
            base1 = OFFSET['COORDINATES'] + 4 * n_coordinates
            base2 = base1 + 4 * n_coordinates
            for i in range(n_coordinates):
                x.append(_get_float(data, base1 + i * 4))
                y.append(_get_float(data, base2 + i * 4))
        else:
            base1 = OFFSET['COORDINATES']
            base2 = base1 + 2 * n_coordinates
            for i in range(n_coordinates):
                xtmp = _get_uint16(data, base1 + i * 2)
                ytmp = _get_uint16(data, base2 + i * 2)
                x.append(left + xtmp)
                y.append(top + ytmp)

        if roi_type == ROI_TYPE['point']:
            roi = {'type': 'point'}
            roi.update(dict(x=x, y=y, n=n_coordinates))
            return roi, roi_props

        if roi_type == ROI_TYPE['polygon']:
            roi = {'type': 'polygon'}

        elif roi_type == ROI_TYPE['freehand']:
            roi = {'type': 'freehand'}
            if subtype == SUBTYPES['ELLIPSE']:
                ex1 = _get_float(data, OFFSET['X1'])
                ey1 = _get_float(data, OFFSET['Y1'])
                ex2 = _get_float(data, OFFSET['X2'])
                ey2 = _get_float(data, OFFSET['Y2'])
                roi['aspect_ratio'] = _get_float(
                    data, OFFSET['ELLIPSE_ASPECT_RATIO'])
                roi.update(dict(ex1=ex1, ey1=ey1, ex2=ex2, ey2=ey2))

                return roi, roi_props

        elif roi_type == ROI_TYPE['traced']:
            roi = {'type': 'traced'}

        elif roi_type == ROI_TYPE['polyline']:
            roi = {'type': 'polyline'}

        elif roi_type == ROI_TYPE['freeline']:
            roi = {'type': 'freeline'}

        elif roi_type == ROI_TYPE['angle']:
            roi = {'type': 'angle'}

        else:
            roi = {'type': 'freeroi'}

        roi.update(dict(x=x, y=y, n=n_coordinates))

        strokeWidth = _get_uint16(data, OFFSET['STROKE_WIDTH'])
        roi.update(dict(width=strokeWidth))

        return roi, roi_props
    else:
        raise UnrecognizedRoiType("Unrecognized ROI specifier: %d" % (roi_type,))


def read_roi_file(fpath: Union[str, zipfile.ZipExtFile]) -> Optional[Dict[str, Union[str, int, float, List[Tuple[float, ...]]]]]:
    """
    Reads an ROI file and returns a dictionary representation of the ROI.

    Args:
        fpath (Union[str, zipfile.ZipExtFile]): Path to the ROI file or a ZipExtFile object.

    Returns:
        Optional[Dict[str, Union[str, int, float, List[Tuple[float, ...]]]]]: Dictionary representation of the ROI.
    """
    if isinstance(fpath, zipfile.ZipExtFile):
        data = fpath.read()
        name = os.path.splitext(os.path.basename(fpath.name))[0]
    elif isinstance(fpath, str):
        with open(fpath, 'rb') as fp:
            data = fp.read()
        name = os.path.splitext(os.path.basename(fpath))[0]
    else:
        logging.error("Can't read {}".format(fpath))
        return None

    logging.debug("Read ROI for \"{}\"".format(name))

    roi, (hdr2Offset, n_coordinates, roi_type, channel, slice, frame, position, version, subtype,
          size) = _extract_basic_roi_data(data)
    roi['name'] = name

    if version >= 218:
        # Not implemented
        # Read stroke width, stroke color and fill color
        pass

    if version >= 218 and subtype == SUBTYPES['TEXT']:
        # Not implemented
        # Read test ROI
        pass

    if version >= 218 and subtype == SUBTYPES['IMAGE']:
        # Not implemented
        # Get image ROI
        pass

    if version >= 224:
        # Not implemented
        # Get ROI properties
        pass

    if version >= 227 and roi['type'] == 'point':
        # Get "point counters" (includes a "counter" and a "position" (slice, i.e. z position)
        tmp = _get_point_counters(data, hdr2Offset, n_coordinates, size)
        if tmp is not None:
            counters, positions = tmp
            if counters:
                roi.update(dict(counters=counters, slices=positions))

    roi['position'] = position
    if channel > 0 or slice > 0 or frame > 0:
        roi['position'] = dict(channel=channel, slice=slice, frame=frame)

    return {name: roi}


def read_roi_zip(zip_path: str) -> List[Optional[Dict[str, Union[str, int, float, List[Tuple[float, ...]]]]]]:
    """
    Reads a zip file containing ROI files and returns a list of dictionary representations of the ROIs.

    Args:
        zip_path (str): Path to the zip file containing ROI files.

    Returns:
        List[Optional[Dict[str, Union[str, int, float, List[Tuple[float, ...]]]]]]: List of dictionary representations of the ROIs.
    """
    rois = []
    zf = zipfile.ZipFile(zip_path)
    name_list = sorted(zf.namelist())
    for n in name_list:
        rois.append(read_roi_file(zf.open(n)))
    return rois


def roi_to_mask(roi_file_dict: Dict[str, Union[str, int, float, List[Tuple[float, ...]]]], roi_folder_name: str, img_size: Tuple[int, int], format_type: str) -> Tuple[np.ndarray, str]:
    """
    Parses single @roi_file_dict into a mask image and corresponding mask file name.

    Args:
        roi_file_dict (Dict[str, Union[str, int, float, List[Tuple[float, ...]]]]): Dictionary for roi file as returned by "read_roi_zip" method.
        roi_folder_name (str): Base name for roi files.
        img_size (Tuple[int, int]): Size of the mask image (width, height).
        format_type (str): Type of format ('camad' or 'whad').

    Returns:
        Tuple[np.ndarray, str]: Mask image and mask image name.
    """
    mask_img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    contour = []
    for pnt_x, pnt_y in zip(roi_file_dict['x'], roi_file_dict['y']):
        pnt_x = max(0, pnt_x)
        pnt_y = max(0, pnt_y)
        contour.append([pnt_x, pnt_y])
    contour = np.array(contour).reshape((-1, 1, 2)).astype(np.int32)

    if format_type == 'whad':
        roi_reg = re.search(r"^(\d+)[-_](W|w|C|c)(\d+)", roi_file_dict['name'])
        roi_index = int(roi_reg.group(1)) - 1
        roi_type = roi_reg.group(2).upper()
        roi_type_index = int(roi_reg.group(3))

        cv2.drawContours(mask_img, [contour], 0, 255, -1)
        if roi_type == "W":
            mask_img = -(mask_img - 255)

        mask_img_name = f"{roi_folder_name}_t{roi_index:02d}_ch00_"
        if roi_type == "W":
            mask_img_name += "mask.png"
        elif roi_type == "C":
            mask_img_name += f"mask_C{roi_type_index:02d}.png"
        else:
            raise Exception(f"Undefined roi type in roi name:\n\t{roi_type}")
    else:
        mask_img = cv2.fillPoly(mask_img, [contour], 255)
        mask_img_name = roi_folder_name + "_mask.png"

    return mask_img, mask_img_name


def roi_zip_to_masks(roi_zip_path: str, format_type: str, out_folder_dir: str = "") -> None:
    """
    Parses roi zip file generated by ImageJ in specified format, and produces
    corresponding mask images at @out_folder_dir if specified or at parent directory of @roi_zip_path.

    Args:
        roi_zip_path (str): Path for zip file of roi files including polygons in ImageJ fashion.
        format_type (str): Type of format ('camad' or 'whad').
        out_folder_dir (str, optional): Target directory for saving resulting mask images. Defaults to "".
    """
    folder_dir, folder_base_name = ntpath.split(roi_zip_path)
    if out_folder_dir:
        folder_dir = out_folder_dir

    if format_type == 'camad':
        folder_base_name = folder_base_name.split("_")[0]
    else:
        folder_base_name = folder_base_name[0:-4]  # Excluding ".zip" part for whad

    roi_list = read_roi_zip(roi_zip_path)
    mask_list = []

    if format_type == 'camad':
        frame_masks = {}
        for roi_file in roi_list:
            roi_key = list(roi_file)[0]
            frame_pos = roi_file[roi_key]['position']
            mask_img, mask_img_name = roi_to_mask(roi_file[roi_key], folder_base_name, (2568, 1912), format_type)
            if frame_pos not in frame_masks:
                frame_masks[frame_pos] = np.zeros_like(mask_img)
            frame_masks[frame_pos] = cv2.bitwise_or(frame_masks[frame_pos], mask_img)

        out_folder_path = os.path.join(folder_dir, folder_base_name)
        os.makedirs(out_folder_path, exist_ok=True)
        for frame_pos, combined_mask in frame_masks.items():
            mask_img_name_with_pos = f"{folder_base_name}_{frame_pos}.png"
            out_file_path = os.path.join(out_folder_path, mask_img_name_with_pos)
            cv2.imwrite(out_file_path, combined_mask)

    else:
        prev_mask_img_name = ""
        for roi_file in roi_list:
            mask_img, mask_img_name = roi_to_mask(roi_file[list(roi_file)[0]], folder_base_name, (1920, 1440),
                                                  format_type)
            if prev_mask_img_name == mask_img_name:
                mask_list[-1][0] = cv2.bitwise_and(mask_list[-1][0], mask_img)
            else:
                mask_list.append([mask_img, mask_img_name])
            prev_mask_img_name = mask_img_name

        out_folder_path = os.path.join(folder_dir, folder_base_name)
        os.makedirs(out_folder_path, exist_ok=True)
        for mask_img, mask_img_name in mask_list:
            out_file_path = os.path.join(out_folder_path, mask_img_name)
            cv2.imwrite(out_file_path, mask_img)

    print(f"Results are saved in: {out_folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parses roi zip file generated by ImageJ in 'wound closing assay annotation' format, and"
                    " produces corresponding mask images. Roi files include polygon points.\nExample usage: "
                    "\n\t python roi_parser.py -src '/home/erdem/Downloads/MCF7-LACZ-p2-25.11.16.zip'")
    parser.add_argument("-src", help="Path for zip file of roi files including polygons in ImageJ"
                                     " fashion.", dest="roi_zip_path", type=str, required=True)
    parser.add_argument("-fmt", help="Data format type ('camad' or 'whad').", dest="format_type", type=str,
                        required=True, choices=['camad', 'whad'])
    parser.add_argument("--out", help="Target directory for saving resulting mask images. (default:"
                                      " @roi_zip_path)", dest="out_folder_dir", type=str, default="",
                        required=False)

    args = parser.parse_args()
    roi_zip_path = args.roi_zip_path
    format_type = args.format_type
    out_folder_dir = args.out_folder_dir
    #
    if not os.path.isfile(roi_zip_path):
        print("The file does not exists: " + roi_zip_path)
        exit()

    roi_zip_to_masks(roi_zip_path, format_type=format_type, out_folder_dir=out_folder_dir)
