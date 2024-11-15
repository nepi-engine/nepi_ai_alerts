#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
#
# This file is part of nepi-engine
# (see https://github.com/nepi-engine).
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#


import os
# ROS namespace setup
#NEPI_BASE_NAMESPACE = '/nepi/s2x/'
#os.environ["ROS_NAMESPACE"] = NEPI_BASE_NAMESPACE[0:-1] # remove to run as automation script
import rospy



import time
import sys
import numpy as np
import cv2
import threading
import copy
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nepi_edge_sdk_base import nepi_ros
from nepi_edge_sdk_base import nepi_save
from nepi_edge_sdk_base import nepi_msg
from nepi_edge_sdk_base import nepi_img 

from std_msgs.msg import UInt8, Int32, Float32, Empty, String, Bool, Header
from sensor_msgs.msg import Image
from nepi_ros_interfaces.msg import BoundingBox, BoundingBoxes, ObjectCount

from nepi_ros_interfaces.srv import ImageClassifierStatusQuery, ImageClassifierStatusQueryRequest

from nepi_app_ai_alerts.msg import AiAlertsStatus, AiAlerts

from nepi_edge_sdk_base.save_data_if import SaveDataIF
from nepi_edge_sdk_base.save_cfg_if import SaveCfgIF

# Do this at the end
#from scipy.signal import find_peaks

#########################################
# Node Class
#########################################

class NepiAiAlertsApp(object):
  AI_MANAGER_NODE_NAME = "ai_detector_mgr"

  #Set Initial Values
  
  
  FACTORY_SENSITIVITY_COUNT = 10
  FACTORY_SENSITIVITY = 1.0

  NONE_CLASSES_DICT = dict()

  data_products = ["Alert_Image","Alerts"]
  
  current_classifier = "None"
  current_classifier_state = "None"
  classes_list = []
  current_classifier_classes = "[]"

  current_image_topic = "None"
  image_source_topic = ""
  img_width = 0
  img_height = 0
  image_sub = None


  last_image_topic = "None"
  
  selected_classes = dict()
 
  alerts_list = []
  alerts_dict = dict()
  active_alert = False


  status_pub = None
  alert_trigger_pub = None
  alerts_pub = None
  image_pub = None
  alerts_boxes_2d_pub = None


  classifier_running = False
  classifier_loading_progress = 0.0
  classifier_threshold = 0.3

  no_object_count = 0

  last_snapshot = time.time()


  reset_image_topic = False
  app_enabled = False
  app_msg = "App not enabled"
  img_acquire = False
  img_msg = None
  last_img_msg = None
  img_lock = threading.Lock()

  
  #######################
  ### Node Initialization
  DEFAULT_NODE_NAME = "app_ai_alerts" # Can be overwitten by luanch command
  def __init__(self):
    #### APP NODE INIT SETUP ####
    nepi_ros.init_node(name= self.DEFAULT_NODE_NAME)
    self.node_name = nepi_ros.get_node_name()
    self.base_namespace = nepi_ros.get_base_namespace()
    nepi_msg.createMsgPublishers(self)
    nepi_msg.publishMsgInfo(self,"Starting Initialization Processes")
    ##############################
    self.ai_mgr_namespace = self.base_namespace + self.AI_MANAGER_NODE_NAME
    
    self.initParamServerValues(do_updates = False)
    self.resetParamServer(do_updates = False)
   


    #self.detection_image_pub = rospy.Publisher("~detection_image",Image,queue_size=1)
    # Setup Node Publishers
    self.status_pub = rospy.Publisher("~status", AiAlertsStatus, queue_size=1, latch=True)
    self.alerts_pub = rospy.Publisher("~alerts", AiAlerts, queue_size=1, latch=True)
    self.alert_trigger_pub = rospy.Publisher("~alert_trigger",Empty,queue_size=1)
    self.image_pub = rospy.Publisher("~alert_image",Image,queue_size=1, latch = True)

    time.sleep(1)


    # Message Image to publish when detector not running
    message = "APP NOT ENABLED"
    cv2_img = nepi_img.create_message_image(message)
    self.app_ne_img = nepi_img.cv2img_to_rosimg(cv2_img)
    self.image_pub.publish(self.app_ne_img)

    message = "WAITING FOR AI DETECTOR TO START"
    cv2_img = nepi_img.create_message_image(message)
    self.classifier_nr_img = nepi_img.cv2img_to_rosimg(cv2_img)


    # Set up save data and save config services ########################################################
    factory_data_rates= {}
    for d in self.data_products:
        factory_data_rates[d] = [0.0, 0.0, 100.0] # Default to 0Hz save rate, set last save = 0.0, max rate = 100.0Hz
    if 'alerts_image' in self.data_products:
        factory_data_rates['alerts_image'] = [1.0, 0.0, 100.0] 
    self.save_data_if = SaveDataIF(data_product_names = self.data_products, factory_data_rate_dict = factory_data_rates)
    # Temp Fix until added as NEPI ROS Node
    self.save_cfg_if = SaveCfgIF(updateParamsCallback=self.initParamServerValues, 
                                 paramsModifiedCallback=self.updateFromParamServer)
    ## App Setup ########################################################
    app_reset_app_sub = rospy.Subscriber('~reset_app', Empty, self.resetAppCb, queue_size = 10)
    self.initParamServerValues(do_updates=False)

    # App Specific Subscribers
    rospy.Subscriber('~publish_status', Empty, self.pubStatusCb, queue_size = 10)
    rospy.Subscriber('~enable_app', Bool, self.appEnableCb, queue_size = 10)
    rospy.Subscriber('~add_all_alert_classes', Empty, self.addAllClassesCb, queue_size = 10)
    rospy.Subscriber('~remove_all_alert_classes', Empty, self.removeAllClassesCb, queue_size = 10)
    rospy.Subscriber('~add_alert_class', String, self.addClassCb, queue_size = 10)
    rospy.Subscriber('~remove_alert_class', String, self.removeClassCb, queue_size = 10)
    #rospy.Subscriber("~set_sensitivity", Float32, self.setSensitivityCb, queue_size = 10)
    rospy.Subscriber('~set_location_str', String, self.setLocationCb, queue_size = 10)
    rospy.Subscriber('~set_snapshot_enable', Bool, self.setSnapshotEnableCb, queue_size = 10)
    rospy.Subscriber("~set_snapshot_delay", Float32, self.setSnapshotDelayCb, queue_size = 10)


    # Get AI Manager Service Call
    AI_MGR_STATUS_SERVICE_NAME = self.ai_mgr_namespace  + "/img_classifier_status_query"
    self.get_ai_mgr_status_service = rospy.ServiceProxy(AI_MGR_STATUS_SERVICE_NAME, ImageClassifierStatusQuery)
    # Start AI Manager Subscribers
    FOUND_OBJECT_TOPIC = self.ai_mgr_namespace  + "/found_object"
    rospy.Subscriber(FOUND_OBJECT_TOPIC, ObjectCount, self.foundObjectCb, queue_size = 1)
    BOUNDING_BOXES_TOPIC = self.ai_mgr_namespace  + "/bounding_boxes"
    rospy.Subscriber(BOUNDING_BOXES_TOPIC, BoundingBoxes, self.objectDetectedCb, queue_size = 1)
    time.sleep(1)

    # Start timed update processes
    nepi_ros.timer(nepi_ros.duration(1), self.updaterCb)

    time.sleep(1)


    ## Initiation Complete
    nepi_msg.publishMsgInfo(self," Initialization Complete")
    self.publish_status()

    # Spin forever (until object is detected)
    nepi_ros.spin()




  #######################
  ### App Config Functions

  def resetAppCb(self,msg):
    self.resetApp()

  def resetApp(self):
    nepi_ros.set_param(self,'~app_enabled',False)
    nepi_ros.set_param(self,'~last_classifier', "")
    nepi_ros.set_param(self,'~selected_classes', [])
    nepi_ros.set_param(self,'~sensitivity', self.FACTORY_SENSITIVITY)
    nepi_ros.set_param(self,'~snapshot_enabled', False)
    nepi_ros.set_param(self,'~snapshot_delay', 5)
    nepi_ros.set_param(self,'~location', "")
    self.publish_status()

  def saveConfigCb(self, msg):  # Just update Class init values. Saving done by Config IF system
    pass # Left empty for sim, Should update from param server

  def setCurrentAsDefault(self):
    self.initParamServerValues(do_updates = False)

  def updateFromParamServer(self):
    #nepi_msg.publishMsgWarn(self,"Debugging: param_dict = " + str(param_dict))
    #Run any functions that need updating on value change
    # Don't need to run any additional functions
    pass

  def initParamServerValues(self,do_updates = True):
      nepi_msg.publishMsgInfo(self," Setting init values to param values")

      self.init_app_enabled = nepi_ros.get_param(self,'~app_enabled',False)
      self.init_last_classifier = nepi_ros.get_param(self,"~last_classifier", "")

      sel_classes = nepi_ros.get_param(self,'~selected_classes', ['All'])
      if 'All' in sel_classes:
        self.addAllClasses()
        time.sleep(1)
      self.init_selected_classes = nepi_ros.get_param(self,'~selected_classes', [])
      
      self.init_sensitivity = nepi_ros.get_param(self,'~sensitivity', self.FACTORY_SENSITIVITY)
      self.init_snapshot_enabled = nepi_ros.get_param(self,'~snapshot_enabled', False)
      self.init_snapshot_delay = nepi_ros.get_param(self,'~snapshot_delay', 5)
      self.init_location = nepi_ros.get_param(self,'~location', "")
      self.resetParamServer(do_updates)

  def resetParamServer(self,do_updates = True):
      nepi_ros.set_param(self,'~app_enabled',self.init_app_enabled)
      nepi_ros.set_param(self,'~last_classiier', self.init_last_classifier)
      nepi_ros.set_param(self,'~selected_classes', self.init_selected_classes)
      nepi_ros.set_param(self,'~sensitivity', self.init_sensitivity)
      nepi_ros.set_param(self,'~snapshot_enabled', self.init_snapshot_enabled)
      nepi_ros.set_param(self,'~snapshot_delay',self.init_snapshot_delay)
      nepi_ros.set_param(self,'~location', self.init_location)
      if do_updates:
          self.updateFromParamServer()
          self.publish_status()


  ###################
  ## Status Publisher
  def publish_status(self):
    status_msg = AiAlertsStatus()

    status_msg.app_enabled = nepi_ros.get_param(self,'~app_enabled',self.init_app_enabled)
    status_msg.app_msg = self.app_msg

    status_msg.location_str = nepi_ros.get_param(self,'~location',self.init_location)
    status_msg.classifier_running = self.classifier_running


    avail_classes = self.classes_list
    #nepi_msg.publishMsgWarn(self," available classes: " + str(avail_classes))
    if len(avail_classes) == 0:
      avail_classes = ["None"]
    avail_classes = sorted(avail_classes)
    status_msg.available_classes_list = avail_classes
    selected_classes = nepi_ros.get_param(self,'~selected_classes', self.init_selected_classes)
    sel_classes = []
    for sel_class in selected_classes:
      if sel_class in avail_classes:
        sel_classes.append(sel_class)
    if len(sel_classes) == 0:
      sel_classes = ['None']
    status_msg.selected_classes_list = (sel_classes)
    nepi_ros.set_param(self,'~selected_classes', sel_classes)
    sensitivity = nepi_ros.get_param(self,'~sensitivity', self.init_sensitivity)
    status_msg.sensitivity = sensitivity
    status_msg.snapshot_enabled = nepi_ros.get_param(self,'~snapshot_enabled', self.init_snapshot_enabled)
    status_msg.snapshot_delay_sec = nepi_ros.get_param(self,'~snapshot_delay', self.init_snapshot_delay)
    status_msg.active_alert = self.active_alert
    self.status_pub.publish(status_msg)

 
  ## Status Publisher
  def publish_alerts(self):
    if self.active_alert == True:
      alerts_msg = AiAlerts()
      stamp = nepi_ros.time_now()
      alerts_msg.header.stamp = stamp
      alerts_msg.date_time_str = nepi_ros.get_datetime_str_from_stamp(stamp)

      alerts_msg.location_str = nepi_ros.get_param(self,'~location',self.init_location)
      alerts_msg.alert_classes_list = self.alerts_list
      self.alerts_pub.publish(alerts_msg)     
    
 

  def updaterCb(self,timer):
    # Save last image topic for next check
    self.last_image_topic = self.current_image_topic
    update_status = False
    app_enabled = nepi_ros.get_param(self,"~app_enabled", self.init_app_enabled)
    app_msg = ""
    if app_enabled == False:
      app_msg += "App not enabled"
      self.alerts_dict = dict()
      self.image_pub.publish(self.app_ne_img)
      if self.image_sub is not None:
        nepi_msg.publishMsgWarn(self," App Disabled, Unsubscribing from Image topic : " + self.last_image_topic)
        self.image_sub.unregister()
        time.sleep(1)
        self.image_sub = None
    else:
      app_msg += "App enabled"
      # Update classifier info
      ai_mgr_status_response = None
      try:
        ai_mgr_status_response = self.get_ai_mgr_status_service()
        #nepi_msg.publishMsgInfo(self," Got classifier status  " + str(ai_mgr_status_response))
      except Exception as e:
        nepi_msg.publishMsgWarn(self,"Failed to call AI MGR STATUS service" + str(e))
        self.classifier_running = False
        nepi_ros.set_param(self,'~last_classiier', "")
        app_msg += ", AI Detector not connected"
      if ai_mgr_status_response != None:
        app_msg += ", AI Detector connected"
        #status_str = str(ai_mgr_status_response)
        #nepi_msg.publishMsgWarn(self," got ai manager status: " + status_str)
        self.current_image_topic = ai_mgr_status_response.selected_img_topic
        self.current_classifier = ai_mgr_status_response.selected_classifier
        self.current_classifier_state = ai_mgr_status_response.classifier_state
        self.classifier_running = self.current_classifier_state == "Running"
        classes_list = ai_mgr_status_response.selected_classifier_classes
        if classes_list != self.classes_list:
          self.classes_list = classes_list
          if len(self.classes_list) > 0:
            cmap = plt.get_cmap('viridis')
            color_list = cmap(np.linspace(0, 1, len(self.classes_list))).tolist()
            rgb_list = []
            for color in color_list:
              rgb = []
              for i in range(3):
                rgb.append(int(color[i]*255))
              rgb_list.append(rgb)
            self.class_color_list = rgb_list
            #nepi_msg.publishMsgWarn(self,self.class_color_list)
          #classes_str = str(self.classes_list)
          #nepi_msg.publishMsgWarn(self," got ai manager status: " + classes_str)
          update_status = True
        selected_classes = nepi_ros.get_param(self,'~selected_classes', self.init_selected_classes)
        last_classifier = nepi_ros.get_param(self,'~last_classiier', self.init_last_classifier)
        if last_classifier != self.current_classifier and self.current_classifier != "None":
          selected_classes = [] # Reset classes to all on new classifier
          update_status = True
        nepi_ros.set_param(self,'~selected_classes', selected_classes)
        nepi_ros.set_param(self,'~last_classiier', self.current_classifier)
        #nepi_msg.publishMsgWarn(self," Got image topics last and current: " + self.last_image_topic + " " + self.current_image_topic)

        # Update Image Topic Subscriber
        if self.classifier_running == False:
          app_msg += ", Classifier not running"
          self.alerts_dict = dict()
        else:
          app_msg += ", Classifier running"
          if (self.last_image_topic != self.current_image_topic) or (self.image_sub == None and self.current_image_topic != "None") or self.reset_image_topic == True:
            self.reset_image_topic = False
            image_topic = nepi_ros.find_topic(self.current_image_topic)
            if image_topic == "":
              nepi_msg.publishMsgWarn(self," Could not find image update topic: " + self.current_image_topic)
              self.image_pub.publish(self.classifier_nr_img)
            else:
              nepi_msg.publishMsgInfo(self," Found detect Image update topic : " + image_topic)
              update_status = True
              if self.image_sub != None:
                nepi_msg.publishMsgWarn(self," Unsubscribing to Image topic : " + self.last_image_topic)
                self.image_sub.unregister()
                time.sleep(1)
                self.image_sub = None
              nepi_msg.publishMsgInfo(self," Subscribing to Image topic : " + image_topic)
              self.image_sub = rospy.Subscriber(image_topic, Image, self.imageCb, queue_size = 1)

          if self.current_image_topic == "None" or self.current_image_topic == "":  # Reset last image topic
            if self.image_sub != None:
              nepi_msg.publishMsgWarn(self," Unsubscribing to Image topic : " + self.current_image_topic)
              self.image_sub.unregister()
              time.sleep(1)
              self.image_sub = None
              update_status = True
              time.sleep(1)
      # Publish warning image if enabled and classifier not running
      if self.classifier_running == False or self.image_sub == None:
        self.classifier_nr_img.header.stamp = nepi_ros.time_now()
        self.image_pub.publish(self.classifier_nr_img)

    self.app_msg = app_msg
    # Publish status if needed
    if update_status == True:
      self.publish_status()
    



  ###################
  ## AI App Callbacks

  def pubStatusCb(self,msg):
    self.publish_status()

  def appEnableCb(self,msg):
    #nepi_msg.publishMsgInfo(self,msg)
    val = msg.data
    nepi_ros.set_param(self,'~app_enabled',val)
    self.publish_status()

  def addAllClassesCb(self,msg):
    self.addAllClasses()
    self.publish_status()

  def addAllClasses(self):
    ##nepi_msg.publishMsgInfo(self,msg)
    nepi_ros.set_param(self,'~selected_classes', self.classes_list)


  def removeAllClassesCb(self,msg):
    ##nepi_msg.publishMsgInfo(self,msg)
    nepi_ros.set_param(self,'~selected_classes',[])
    self.alert_dict = dict()
    self.publish_status()

  def addClassCb(self,msg):
    ##nepi_msg.publishMsgInfo(self,msg)
    class_name = msg.data
    if class_name in self.classes_list:
      sel_classes = nepi_ros.get_param(self,'~selected_classes', self.init_selected_classes)
      if class_name not in sel_classes:
        sel_classes.append(class_name)
      nepi_ros.set_param(self,'~selected_classes', sel_classes)
    self.publish_status()

  def removeClassCb(self,msg):
    ##nepi_msg.publishMsgInfo(self,msg)
    class_name = msg.data
    sel_classes = nepi_ros.get_param(self,'~selected_classes', self.init_selected_classes)
    if class_name in sel_classes:
      sel_classes.remove(class_name)
      nepi_ros.set_param(self,'~selected_classes', sel_classes)
      if class_name in self.alerts_dict.keys():
        self.alerts_dict[class_name] = 0
    self.publish_status()

  def setSensitivityCb(self,msg):
    nepi_msg.publishMsgInfo(self,msg)
    val = msg.data
    if val >= 0 and val <= 1:
      nepi_ros.set_param(self,'~sensitivity',val)
    self.publish_status()

  def setSnapshotEnableCb(self,msg):
    #nepi_msg.publishMsgInfo(self,msg)
    val = msg.data
    nepi_ros.set_param(self,'~snapshot_enabled',val)
    self.publish_status()

  def setSnapshotDelayCb(self,msg):
    #nepi_msg.publishMsgInfo(self,msg)
    val = msg.data
    if val > 0 :
      nepi_ros.set_param(self,'~snapshot_delay',val)
    self.publish_status()

  def setLocationCb(self,msg):
    ##nepi_msg.publishMsgInfo(self,msg)
    location_str = msg.data
    nepi_ros.set_param(self,'~location', location_str)
    self.publish_status()

  

  #######################
  ### AI Magnager Callbacks


  ### If object(s) detected, save bounding box info to global
  def objectDetectedCb(self,bounding_boxes_msg):
    app_enabled = nepi_ros.get_param(self,"~app_enabled", self.init_app_enabled)
    ros_timestamp = bounding_boxes_msg.header.stamp

    if app_enabled == False:
      self.alerts_list = []
    else:
      alerts_list = []
      alert_boxes = []
      sel_classes = nepi_ros.get_param(self,'~selected_classes', self.init_selected_classes)
      for box in bounding_boxes_msg.bounding_boxes:
        if box.Class in sel_classes:
          alert_boxes.append(box)
          if box.Class not in alerts_list:
            alerts_list.append(box.Class)
      if len(alerts_list) > 0:
        self.active_alert = True
        self.alerts_list = alerts_list
        # create save dict
        alerts_dict = dict()
        alerts_dict['timestamp'] = nepi_ros.get_datetime_str_from_stamp(ros_timestamp)
        alerts_dict['location'] = nepi_ros.get_param(self,'~location',self.init_location)
        alerts_dict['alert_classes_list'] = self.alerts_list
        nepi_save.save_dict2file(self,'Alerts',alerts_dict,ros_timestamp,save_check = True)
        self.publish_alerts()
        self.alert_trigger_pub.publish(Empty())
      else:
        self.active_alert = False



      # Process image 
      img_in_msg = None
      self.img_lock.acquire()
      img_in_msg = copy.deepcopy(self.img_msg) 
      self.img_msg = None # Clear the last image        
      self.img_lock.release()
      
      if len(alert_boxes) == 0:
        if img_in_msg is not None and not nepi_ros.is_shutdown():
          self.image_pub.publish(img_in_msg)
      else:
        if img_in_msg is not None:
          current_image_header = img_in_msg.header
          ros_timestamp = img_in_msg.header.stamp     
          cv2_img = nepi_img.rosimg_to_cv2img(img_in_msg).astype(np.uint8)
          cv2_shape = cv2_img.shape
          self.img_width = cv2_shape[1] 
          self.img_height = cv2_shape[0]   
        
          for box in alert_boxes:
            #nepi_msg.publishMsgWarn(self," Box: " + str(box))
            class_name = box.Class
            if class_name in alerts_list:
              [xmin,xmax,ymin,ymax] = [box.xmin,box.xmax,box.ymin,box.ymax]
              start_point = (xmin, ymin)
              end_point = (xmax, ymax)
              class_name = class_name
              class_color = (0,0,255)
              line_thickness = 2
              cv2.rectangle(cv2_img, start_point, end_point, class_color, thickness=line_thickness)

            # Publish new image to ros
            if not nepi_ros.is_shutdown() and self.image_pub is not None: #and has_subscribers:
                #Convert OpenCV image to ROS image
                cv2_shape = cv2_img.shape
                if  cv2_shape[2] == 3:
                  encode = 'bgr8'
                else:
                  encode = 'mono8'
                img_out_msg = nepi_img.cv2img_to_rosimg(cv2_img, encoding=encode)
                self.image_pub.publish(img_out_msg)
            # Save Data if Time
            nepi_save.save_img2file(self,'alert_image',cv2_img,ros_timestamp,save_check = True)
    self.publish_status()

  def imageCb(self,image_msg):    
      self.img_lock.acquire()
      self.img_msg = copy.deepcopy(self.last_img_msg)
      self.img_lock.release()
      self.last_img_msg = copy.deepcopy(image_msg)



  ### Monitor Output of AI model to clear detection status
  def foundObjectCb(self,found_obj_msg):
    app_enabled = nepi_ros.get_param(self,"~app_enabled", self.init_app_enabled)

    #Clean Up Detection and Alert data
    if found_obj_msg.count == 0:
      #  Run App Process
      if app_enabled == True:
        # publish current image msg with no overlay
        img_in_msg = None
        self.img_lock.acquire()
        img_in_msg = copy.deepcopy(self.img_msg) 
        self.img_msg = None # Clear the last image        
        self.img_lock.release()
        if not nepi_ros.is_shutdown() and self.image_pub is not None and img_in_msg is not None:
          self.image_pub.publish(img_in_msg)

        self.alerts_list = []
        self.alerts_dict = dict()
        self.active_alert = False
        
        self.publish_status()







    


      
               
    
  #######################
  # Node Cleanup Function
  
  def cleanup_actions(self):
    nepi_msg.publishMsgInfo(self," Shutting down: Executing script cleanup actions")


#########################################
# Main
#########################################
if __name__ == '__main__':
  NepiAiAlertsApp()







