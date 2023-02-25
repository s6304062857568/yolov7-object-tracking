import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn

from zone_detection import find_zone_by_position, draw_ROI

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
                time_synchronized, TracedModel
from utils.download_weights import download

from skimage.metrics import structural_similarity

#For SORT tracking
import skimage
from sort import *

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id= opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.colored_trk, opt.save_bbox_dim, opt.save_with_object_id
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 9 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    #......................... 
    
    
    #........Rand Color for every trk.......
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................
   

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    id_zone_frame = {} # Declare variables
    before = None # image blueprint
    points = np.array([[775,210], [890,240], [785,515], [612,460]]) # point for prevent reflect
    
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            if frame_idx == 0:
              before = im0.copy()
              # Draw a filled white polygon
              cv2.fillPoly(before, pts=[points], color=(255, 255, 255))

            # fill white background
            cv2.fillPoly(im0, pts=[points], color=(255, 255, 255))

            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
              
            # org
            org = (10, 50)
              
            # fontScale
            fontScale = 1
              
            # Blue color in BGR
            color = (255, 0, 0)
              
            # Line thickness of 2 px
            thickness = 2

            cv2.putText(im0, str(frame_idx+1), org, font, fontScale, color, thickness, cv2.LINE_AA)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            txt_path = str(save_dir / 'labels' / p.stem)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()

                txt_str = ""

                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    #print('\n', bbox_xyxy, '\n')
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    #draw_boxes(im0, bbox_xyxy, identities, categories, names, save_with_object_id, txt_path)

                    # -------------- Start : Custom code -------------- #
                    for i, box in enumerate(bbox_xyxy):
                      # to MOT format
                      bbox_left = int(box[0])
                      bbox_top = int(box[1])
                      bbox_w = int(box[2] - box[0])
                      bbox_h = int(box[3] - box[1])
                      bbox_bx = int(box[2])
                      bbox_by = int(box[3])

                      x1, y1, x2, y2 = [int(abs(i)) for i in box]
                      
                      # -------------- Start : Find foot position -------------- #
                      after = im0.copy()
                      # Draw a filled white polygon
                      cv2.fillPoly(after, pts=[points], color=(255, 255, 255))

                      after_sliced = after[int(y1+(y2-y1)-50):int(y2), int((x1)):int(x2)]
                      before_sliced = before[int(y1+(y2-y1)-50):int(y2), int((x1)):int(x2)]

                      # Convert images to grayscale
                      before_gray = cv2.cvtColor(before_sliced, cv2.COLOR_BGR2GRAY)
                      after_gray = cv2.cvtColor(after_sliced, cv2.COLOR_BGR2GRAY)
                      
                      # Compute SSIM between the two images
                      (score, diff) = structural_similarity(before_gray, after_gray, full=True)
                      #print("Image Similarity: {:.4f}%".format(score * 100))

                      # The diff image contains the actual image differences between the two images
                      # and is represented as a floating point data type in the range [0,1] 
                      # so we must convert the array to 8-bit unsigned integers in the range
                      # [0,255] before we can use it with OpenCV
                      diff = (diff * 255).astype("uint8")
                      diff_box = cv2.merge([diff, diff, diff])

                      # Threshold the difference image, followed by finding contours to
                      # obtain the regions of the two input images that differ
                      thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                      contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                      contours = contours[0] if len(contours) == 2 else contours[1]

                      biggest_area = 0
                      bb_box = [] # fix here
                      for c in contours:
                          area = cv2.contourArea(c)
                          #print('area:',area)

                          if int(area) > 80:
                              x,y,w,h = cv2.boundingRect(c)
                              # cal biggest area
                              if area > biggest_area:
                                biggest_area = area
                                bb_box = [x,y,w,h]

                      #cv2.rectangle(im0, (x1, y1), (x2, y2), (255,0,20), 2)
                      if len(bb_box) > 0:
                        x,y,w,h = bb_box
                        #position_roi = (int(x1+x+(w/2)), int(y2-(h/2))) # centroid of biggest contourArea
                        #position_roi = (int(x1+x+(w/2)), int(y2-5)) # bottom centroid of biggest contourArea
                        position_roi = (int(x1+x+(w/2)), int(y2-(h/2))) # bottom centroid of biggest contourArea
                        cv2.circle(im0, position_roi, 2, [0,69,255], 2) # position of ROI

                        cv2.rectangle(im0, (x1+x+4, y2-y-h+2), (x1+x+w-4, y2-y-4), (0,215,255), 1) # Bounding box biggest
                        cv2.rectangle(im0, (x1+2, y2-50), (x2-2, y2-2), (0,0,255), 1) # Bounding box selected
                      else:
                        position_roi = (int((box[0]+box[2])/2), int(box[3]-5)) # centroid of bounding box
                        cv2.circle(im0, position_roi, 2, [255,255,255], 2) # position of ROI

                      # -------------- End : Find foot position -------------- #

                      cat = int(categories[i]) if categories is not None else 0
                      id = int(identities[i]) if identities is not None else 0
                      data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
                      roi = (int((box[0]+box[2])/2),(int(box[3]-10)))
                      
                      zone = find_zone_by_position(position_roi)

                      label = str(id) + ":"+ names[cat] + ":" + zone
                      (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                      cv2.rectangle(im0, (x1, y1), (x2, y2), (255,0,20), 2)
                      cv2.rectangle(im0, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
                      cv2.putText(im0, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, [255, 255, 255], 1)
                      #cv2.circle(im0, roi, 3, [0,69,255], 3) # position of ROI

                      # Write MOT compliant results to file
                      with open(txt_path + '.txt', 'a') as f:
                          f.write(('%g ' * 12 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                          bbox_top, bbox_bx, bbox_by, -1, -1, -1, i, position_roi[0], position_roi[1]))
                                                          
                      # save detail each id -> id, zone, frame
                      if id in id_zone_frame:
                        zone_dict = id_zone_frame[id]
                        if zone in zone_dict:
                          frame_set = zone_dict[zone]
                          frame_set.append(frame_idx)
                        elif zone != '':
                          frame_set = [frame_idx]
                          zone_dict[zone] = frame_set
                          id_zone_frame[id][zone] = zone_dict[zone]
                      elif zone != '':
                        frame_set = [frame_idx]
                        zone_dict = {zone : frame_set}
                        id_zone_frame[id] = zone_dict
                      # -------------- End : Custom code -------------- #
                      
                #........................................................
                
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
        
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                  cv2.destroyAllWindows()
                  raise StopIteration

            # Save results (image with detections)
            if save_img:
                draw_ROI(im0)

                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img or save_with_object_id:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    print(id_zone_frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--save-bbox-dim', action='store_true', help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')

    parser.set_defaults(download=True)
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    if opt.download and not os.path.exists(str(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['/content/yolov7-object-tracking/yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
