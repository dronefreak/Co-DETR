#!/usr/bin/env python
"""Unified inference CLI for Co-DETR: run a trained checkpoint on a single
image, a folder of images, or a video, and write annotated outputs (and,
optionally, raw detections as JSON) without having to pick between the
various demo scripts by hand.

For simple one-off exploration you can also use `demo/image_demo.py` /
`demo/video_demo.py` directly -- this script is meant to be the "just works"
end-to-end entry point for images *and* video, with consistent flags and
optional structured (JSON) output for downstream consumption (e.g. feeding
into an eval/analysis pipeline).

Examples
--------
Single image:
    python tools/inference.py \\
        --config projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py \\
        --checkpoint checkpoints/co_dino_5scale_r50_1x_coco.pth \\
        --input demo/demo.jpg --out-dir outputs/

Folder of images, saving detections as JSON too:
    python tools/inference.py \\
        --config <config>.py --checkpoint <ckpt>.pth \\
        --input path/to/image_folder --out-dir outputs/ --save-json

Video:
    python tools/inference.py \\
        --config <config>.py --checkpoint <ckpt>.pth \\
        --input path/to/video.mp4 --out-dir outputs/
"""
import argparse
import json
import os
import os.path as osp

import cv2
import mmcv

from mmdet.apis import inference_detector, init_detector
from projects import *  # noqa: F401,F403  registers CoDETR / ViT / SFP / CoDINOHead

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def parse_args():
    parser = argparse.ArgumentParser(description='Co-DETR inference on images/video')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file path')
    parser.add_argument(
        '--input',
        required=True,
        help='path to a single image, a directory of images, or a video '
        'file. The input type is auto-detected from the path/extension.')
    parser.add_argument(
        '--out-dir',
        default='outputs',
        help='directory to write annotated image(s)/video and (optionally) '
        'detections.json into (default: outputs/)')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='also dump raw per-image/frame detections '
        '(class, bbox, score) to <out-dir>/detections.json')
    parser.add_argument(
        '--show', action='store_true',
        help='display results in a window while processing (images or video)')
    return parser.parse_args()


def result_to_dets(model, result, score_thr):
    """Convert an mmdet inference result into a list of plain-dict
    detections: [{'category': str, 'bbox': [x1, y1, x2, y2], 'score': float}, ...]
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


def run_on_video(model, video_path, out_dir, score_thr, show, save_json):
    video_reader = mmcv.VideoReader(video_path)
    out_file = osp.join(out_dir, osp.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        out_file, fourcc, video_reader.fps,
        (video_reader.width, video_reader.height))

    all_detections = []
    for frame_idx, frame in enumerate(
            mmcv.track_iter_progress(video_reader)):
        result = inference_detector(model, frame)
        vis_frame = model.show_result(frame, result, score_thr=score_thr)
        if show:
            cv2.namedWindow('inference', 0)
            mmcv.imshow(vis_frame, 'inference', 1)
        video_writer.write(vis_frame)
        if save_json:
            all_detections.append({
                'frame': frame_idx,
                'detections': result_to_dets(model, result, score_thr),
            })

    video_writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f'[inference] wrote annotated video to {out_file}')
    return all_detections


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)

    model = init_detector(args.config, args.checkpoint, device=args.device)
    print(f'[inference] model built, {len(model.CLASSES)} classes')

    if osp.isdir(args.input):
        image_paths = sorted(
            osp.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith(IMAGE_EXTENSIONS))
        if not image_paths:
            raise ValueError(f'No images with extensions {IMAGE_EXTENSIONS} '
                             f'found in {args.input}')
        detections = run_on_images(model, image_paths, args.out_dir,
                                   args.score_thr, args.show, args.save_json)
    elif args.input.lower().endswith(VIDEO_EXTENSIONS):
        detections = run_on_video(model, args.input, args.out_dir,
                                  args.score_thr, args.show, args.save_json)
    elif args.input.lower().endswith(IMAGE_EXTENSIONS):
        detections = run_on_images(model, [args.input], args.out_dir,
                                   args.score_thr, args.show, args.save_json)
    else:
        raise ValueError(
            f'Could not infer input type for {args.input!r}: expected a '
            f'directory, or a file with one of these extensions: '
            f'images={IMAGE_EXTENSIONS}, videos={VIDEO_EXTENSIONS}')

    if args.save_json:
        json_path = osp.join(args.out_dir, 'detections.json')
        with open(json_path, 'w') as f:
            json.dump(detections, f, indent=2)
        print(f'[inference] wrote detections to {json_path}')


if __name__ == '__main__':
    main()
