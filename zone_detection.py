import cv2
from shapely.geometry import *

def find_zone(bboxes):
  MAX_H = 720
  MAX_W = 1280
  zones = []

  ice_polygon_xy = [(830,MAX_H - 80),(735,MAX_H - 235),(840,MAX_H - 265),(910,MAX_H - 105)]
  water_polygon_xy = [(740,MAX_H - 55),(625,MAX_H - 200),(735,MAX_H - 235),(830,MAX_H - 80)]
  alcohol_polygon_xy = [(625,MAX_H - 200),(415,MAX_H - 467),(530,MAX_H - 525),(735,MAX_H - 235)]
  popsicle_polygon_xy = [(735,MAX_H - 235),(530,MAX_H - 525),(675,MAX_H - 590),(840,MAX_H - 265)]
  counter_polygon_xy = [(415,MAX_H - 467),(235,MAX_H - 718),(460,MAX_H - 718),(565,MAX_H - 540)]
  snack_polygon_xy = [(565,MAX_H - 540),(460,MAX_H - 718),(725,MAX_H - 718),(760,MAX_H - 630)]

  zones.append(snack_polygon_xy)
  zones.append(counter_polygon_xy)
  zones.append(popsicle_polygon_xy)
  zones.append(ice_polygon_xy)
  zones.append(water_polygon_xy)
  zones.append(alcohol_polygon_xy)

  posistion_roi = (int(bboxes[0]) + ((bboxes[2])/2), MAX_H - int((bboxes[1])+(bboxes[3])) + 10)
  index = 0
  for zone in zones:
      polygon_zone_shape = Polygon(zone)
      #intersection = polygon_zone_shape.intersection(pol)
      #intersection_ratio = intersection.area / pol.area
      if (polygon_zone_shape.intersects(Point(posistion_roi))):
        print('zone',number_to_string(index))
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
  cv2.line(im0,(830,80),(910,105),(0, 125, 255),2) # โซนน้ำแข็ง เส้นบน]
  cv2.line(im0,(735,235),(840,265),(0, 125, 255),2) # โซนน้ำแข็ง เส้นล่าง
  cv2.line(im0,(830,80),(735,235),(0, 125, 255),2) # โซนน้ำแข็ง เส้นซ้าย(xบน,xล่าง)
  cv2.line(im0,(910,105),(840,265),(0, 125, 255),2) # โซนน้ำแข็ง เส้นขวา(yบน,yล่าง)

  cv2.line(im0,(735,235),(840,265),(0, 255, 0),2) # โซนไอติม เส้นบน
  cv2.line(im0,(530,525),(675,590),(0, 255, 0),2) # โซนไอติม เส้นล่าง ซ้าย - ขวา
  cv2.line(im0,(735,235),(530,525),(0, 255, 0),2) # โซนไอติม เส้นซ้าย(xบน,xล่าง)
  cv2.line(im0,(840,265),(675,590),(0, 255, 0),2) # โซนไอติม เส้นขวา(yบน,yล่าง)

  # Alcohol
  cv2.line(im0,(625,200),(735,235),(125, 0, 255),2) # Alcohol เส้นบน
  cv2.line(im0,(625,200),(415,467),(125, 0, 255),2) # Alcohol เส้นซ้าย
  cv2.line(im0,(415,467),(530,525),(125, 0, 255),2) # Alcohol เส้นล่าง
  cv2.line(im0,(735,235),(530,525),(125, 0, 255),2) # Alcohol เส้นขวา บน-ล่าง

  # Water
  cv2.line(im0,(740,55),(830,80),(125, 0, 80),2) # Water เส้นบน
  cv2.line(im0,(740,55),(625,200),(125, 0, 80),2) # Water เส้นซ้าย
  cv2.line(im0,(625,200),(735,235),(125, 0, 80),2) # Water เส้นล่าง
  cv2.line(im0,(830,80),(735,235),(125, 0, 80),2) # Water เส้นขวา บน-ล่าง

  cv2.line(im0,(415,467),(565,540),(0, 255, 0),2) # Counter เส้นบน
  cv2.line(im0,(235,718),(415,467),(0, 255, 0),2) # Counter เส้นซ้าย
  cv2.line(im0,(235,718),(460,718),(0, 255, 0),2) # Counter เส้นล่าง
  cv2.line(im0,(565,540),(460,718),(0, 255, 0),2) # Counter เส้นขวา บน-ล่าง

  cv2.line(im0,(565,540),(760,630),(255, 125, 0),2) # Snack เส้นบน
  cv2.line(im0,(460,718),(725,718),(255, 125, 0),2) # Snack เส้นซ้าย
  cv2.line(im0,(565,540),(460,718),(255, 125, 0),2) # Snack เส้นล่าง
  cv2.line(im0,(760,630),(725,718),(255, 125, 0),2) # Snack เส้นขวา
  
