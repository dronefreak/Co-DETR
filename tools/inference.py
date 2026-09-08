#!/usr/bin/env python
"""Unified inference CLI for Co-DETR.

One entry point for every input type -- a single image, a folder of images,
a video file, or a live webcam -- with consistent flags and optional
structured (JSON) output for downstream consumption.

The input type is auto-detected from ``--input``:

* an existing file with an image extension  -> single-image mode
* an existing directory                     -> folder mode (all images in it)
* an existing file with a video extension   -> video-file mode
* ``webcam`` / ``webcam:<id>`` / a bare int -> live-camera mode (id, default 0)

Examples
--------
Single image::

    python tools/inference.py \\
        --config projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py \\
        --checkpoint checkpoints/co_deformable_detr_r50_1x_coco.pth \\
        --input assets/cats_remotes.jpg --out-dir inference/demo

Folder of images, also dumping detections as JSON::

    python tools/inference.py --config <cfg>.py --checkpoint <ckpt>.pth \\
        --input assets/ --out-dir inference/demo --save-json

Video file (write an annotated .mp4, stop after 100 frames)::

    python tools/inference.py --config <cfg>.py --checkpoint <ckpt>.pth \\
        --input demo/demo.mp4 --out-dir inference/demo --max-frames 100

Webcam (show a live window; press q/Esc to quit; --record also saves it)::

    python tools/inference.py --config <cfg>.py --checkpoint <ckpt>.pth \\
        --input webcam --show --record --out-dir inference/demo
"""
import argparse
import json
import os
import os.path as osp
import time

import cv2
import mmcv

from mmdet.apis import inference_detector, init_detector
from projects import *  # noqa: F401,F403  registers CoDETR / ViT / SFP / CoDINOHead

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Co-DETR inference on an image / folder / video / webcam')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file path')
    parser.add_argument(
        '--input',
        required=True,
        help="a single image, a directory of images, a video file, or "
        "'webcam' / 'webcam:<id>' / a bare integer for a live camera. "
        "The mode is auto-detected from the value.")
    parser.add_argument(
        '--out-dir',
        default='outputs',
        help='directory to write annotated image(s)/video and (optionally) '
        'detections.json into (default: outputs/)')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference '
        "(e.g. 'cuda:0', 'cpu')")
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='also dump raw per-image/frame detections '
        '(class, bbox, score) to <out-dir>/detections.json')
    parser.add_argument(
        '--max-frames',
        type=int,
        default=0,
        help='video/webcam only: stop after this many frames (0 = no limit)')
    parser.add_argument(
        '--record',
        action='store_true',
        help='webcam only: also save the annotated stream to '
        '<out-dir>/webcam_<timestamp>.mp4 (video-file mode always writes one)')
    parser.add_argument(
        '--show', action='store_true',
        help='display results in a window while processing (needs a display; '
        'for webcam this is how you press q/Esc to quit)')
    return parser.parse_args()


def parse_input_spec(value):
    """Classify ``--input`` into ('image'|'folder'|'video'|'webcam', target)."""
    if value.lower() in ('webcam', 'cam', 'camera'):
        return 'webcam', 0
    if value.lower().startswith(('webcam:', 'cam:', 'camera:')):
        return 'webcam', int(value.split(':', 1)[1])
    if value.isdigit():
        return 'webcam', int(value)
    if osp.isdir(value):
        return 'folder', value
    if osp.isfile(value):
        low = value.lower()
        if low.endswith(IMAGE_EXTENSIONS):
            return 'image', value
        if low.endswith(VIDEO_EXTENSIONS):
            return 'video', value
        raise ValueError(
            f'{value!r} is a file but not a recognized image '
            f'({IMAGE_EXTENSIONS}) or video ({VIDEO_EXTENSIONS}) type.')
    raise ValueError(
        f'Could not resolve --input {value!r}: not a webcam spec, and no such '
        f'file or directory.')


def result_to_dets(model, result, score_thr):
    """Convert an mmdet inference result into a list of plain-dict detections:
    [{'category': str, 'bbox': [x1, y1, x2, y2], 'score': float}, ...]
    """
    bbox_result = result[0] if isinstance(result, tuple) else result
    dets = []
    for cls_idx, bboxes in enumerate(bbox_result):
        for bbox in bboxes:
            score = float(bbox[4])
            if score < score_thr:
                continue
            dets.append({
                'category': model.CLASSES[cls_idx],
                'bbox': [float(x) for x in bbox[:4]],
                'score': score,
            })
    return dets


def run_on_images(model, image_paths, out_dir, score_thr, show, save_json):
    all_detections = {}
    for img_path in mmcv.track_iter_progress(image_paths):
        result = inference_detector(model, img_path)
        out_file = osp.join(out_dir, osp.basename(img_path))
        model.show_result(
            img_path, result, score_thr=score_thr, out_file=out_file,
            show=show)
        if save_json:
            all_detections[osp.basename(img_path)] = result_to_dets(
                model, result, score_thr)
    print(f'[inference] wrote {len(image_paths)} annotated image(s) to {out_dir}')
    return all_detections


def run_on_stream(model, source, out_dir, score_thr, show, save_json,
                  max_frames, record, is_webcam):
    """Shared frame loop for both video files and webcams.

    ``source`` is a path (video file) or an int camera id (webcam).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        what = f'camera id {source}' if is_webcam else f'video {source!r}'
        raise RuntimeError(
            f'[inference] could not open {what}. '
            + ('Is a camera connected / not in use by another app?'
               if is_webcam else 'Is the file a readable video?'))

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not fps or fps != fps:  # 0 or NaN (common on webcams)
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    writer = None
    if record or not is_webcam:
        if is_webcam:
            out_file = osp.join(
                out_dir, time.strftime('webcam_%Y%m%d_%H%M%S.mp4'))
        else:
            stem = osp.splitext(osp.basename(source))[0]
            out_file = osp.join(out_dir, f'{stem}_pred.mp4')
        writer = cv2.VideoWriter(
            out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    if is_webcam:
        print('[inference] webcam live. Press q / Q / Esc in the window to '
              'stop' + ('' if show else ' (or Ctrl-C here, since --show is off)')
              + '.')

    all_detections = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = inference_detector(model, frame)
            vis = model.show_result(frame, result, score_thr=score_thr)
            if show:
                cv2.imshow('co-detr inference', vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
            if writer is not None:
                writer.write(vis)
            if save_json:
                all_detections.append({
                    'frame': frame_idx,
                    'detections': result_to_dets(model, result, score_thr),
                })
            frame_idx += 1
            if not is_webcam and total:
                if frame_idx % 10 == 0 or frame_idx == total:
                    print(f'\r[inference] frame {frame_idx}/{total}', end='')
            elif frame_idx % 30 == 0:
                print(f'\r[inference] frame {frame_idx}', end='')
            if max_frames and frame_idx >= max_frames:
                break
    except KeyboardInterrupt:
        print('\n[inference] interrupted, finalizing outputs...')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print()
    if writer is not None:
        print(f'[inference] wrote {frame_idx} annotated frame(s) to {out_file}')
    else:
        print(f'[inference] processed {frame_idx} frame(s)')
    return all_detections


def main():
    args = parse_args()
    mode, target = parse_input_spec(args.input)
    mmcv.mkdir_or_exist(args.out_dir)

    model = init_detector(args.config, args.checkpoint, device=args.device)
    print(f'[inference] model built ({mode} mode), {len(model.CLASSES)} classes')

    if mode == 'folder':
        image_paths = sorted(
            osp.join(target, f) for f in os.listdir(target)
            if f.lower().endswith(IMAGE_EXTENSIONS))
        if not image_paths:
            raise ValueError(f'No images with extensions {IMAGE_EXTENSIONS} '
                             f'found in {target}')
        detections = run_on_images(model, image_paths, args.out_dir,
                                   args.score_thr, args.show, args.save_json)
    elif mode == 'image':
        detections = run_on_images(model, [target], args.out_dir,
                                   args.score_thr, args.show, args.save_json)
    else:  # 'video' or 'webcam'
        detections = run_on_stream(
            model, target, args.out_dir, args.score_thr, args.show,
            args.save_json, args.max_frames, args.record,
            is_webcam=(mode == 'webcam'))

    if args.save_json:
        json_path = osp.join(args.out_dir, 'detections.json')
        with open(json_path, 'w') as f:
            json.dump(detections, f, indent=2)
        print(f'[inference] wrote detections to {json_path}')


if __name__ == '__main__':
    import sys
    try:
        main()
    except (RuntimeError, ValueError) as e:
        # expected, user-facing failures (bad --input, camera/file won't open):
        # print the message, skip the traceback.
        print(str(e), file=sys.stderr)
        sys.exit(1)
