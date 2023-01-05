import cv2
import numpy as np
from shapely.geometry import *

def find_zone_by_position(position):
  zones = []

  ice_polygon_xy = ((710, 235), (793, 110), (855, 135), (780, 265))
  water_polygon_xy = ((630, 207), (755, 55), (816, 75), (710, 235))
  alcohol_polygon_xy = ((415, 489), (630, 207), (710, 235), (505, 527))
  popsicle_polygon_xy = ((505, 527), (710, 235), (780, 265), (585, 560))
  counter_polygon_xy = ((460, 508), (585, 560), (480, 718), (278, 718))
  snack_polygon_xy = ((567, 588), (735, 655), (705, 718), (480, 718))

  zones.append(snack_polygon_xy)
  zones.append(counter_polygon_xy)
  zones.append(popsicle_polygon_xy)
  zones.append(ice_polygon_xy)
  zones.append(water_polygon_xy)
  zones.append(alcohol_polygon_xy)

  index = 0
  for zone in zones:
      polygon_zone_shape = Polygon(zone)
      if (polygon_zone_shape.intersects(Point(position))):
        #print('zone',number_to_string(index))
        return number_to_string(index)
      
      index += 1

  return ""

def number_to_string(argument):
    if argument == 0:
        return "snack"
    if argument == 1:
        return "counter"
    if argument == 2:
        return "popsicle"
    if argument == 3:
        return "ice"
    if argument == 4:
        return "water"
    if argument == 5:
        return "alcohol"
    
    return "undefined"

def draw_ROI(im0):
  # Reference : https://www.geeksforgeeks.org/python-opencv-cv2-polylines-method/
  pts_counter = np.array([[460, 508], [585, 560], [480, 718], [278, 718]], np.int32)
  pts_counter = pts_counter.reshape((-1, 1, 2))

  pts_snack = np.array([[567, 588], [735, 655], [705, 718], [480, 718]], np.int32)
  pts_snack = pts_snack.reshape((-1, 1, 2))

  pts_popsicle = np.array([[505, 527], [710, 235], [780, 265], [585, 560]], np.int32)
  pts_popsicle = pts_popsicle.reshape((-1, 1, 2))

  pts_alcohol = np.array([[415, 489], [630, 207], [710, 235], [505, 527]], np.int32)
  pts_alcohol = pts_alcohol.reshape((-1, 1, 2))

  pts_water = np.array([[630, 207], [755, 55], [816, 75], [710, 235]], np.int32)
  pts_water = pts_water.reshape((-1, 1, 2))

  pts_ice = np.array([[710, 235], [793, 110], [855, 135], [780, 265]], np.int32)
  pts_ice = pts_ice.reshape((-1, 1, 2))

  isClosed = True
    
  # Blue color in BGR
  color = (255, 0, 0)
    
  # Line thickness of  px
  thickness = 1

  cv2.polylines(im0, [pts_counter], isClosed, (0, 255, 0), thickness)
  cv2.polylines(im0, [pts_snack], isClosed, (255, 125, 0), thickness)
  cv2.polylines(im0, [pts_popsicle], isClosed, (0, 255, 0), thickness)
  cv2.polylines(im0, [pts_alcohol], isClosed, (125, 0, 255), thickness)
  cv2.polylines(im0, [pts_water], isClosed, (125, 0, 80), thickness)
  cv2.polylines(im0, [pts_ice], isClosed, (0, 125, 255), thickness)
