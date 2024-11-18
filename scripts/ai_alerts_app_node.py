#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus <https://www.numurus.com>.
#
# This file is part of nepi applications (nepi_apps) repo
# (see https://https://github.com/nepi-engine/nepi_apps)
#
# License: nepi applications are licensed under the "Numurus Software License", 
# which can be found at: <https://numurus.com/wp-content/uploads/Numurus-Software-License-Terms.pdf>
#
# Redistributions in source code must retain this top-level comment block.
# Plagiarizing this software to sidestep the license obligations is illegal.
#
# Contact Information:
# ====================
# - mailto:nepi@numurus.com
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

  UDATE_PROCESS_DELAY = 1
  IMG_PUB_PROCESS_DELAY = 0.2

  FACTORY_ALERT_DELAY = 3.0
  FACTORY_CLEAR_DELAY = 2.0
  FACTORY_TRIGGER_DELAY = 10.0


  NONE_CLASSES_DICT = dict()

  data_products = ["alert_image","alert_data"]
  
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
 
  alert_boxes = []
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

  reset_image_topic = False
  app_enabled = False
  app_msg = "App not enabled"
  img_acquire = False
  img_msg = None
  last_img_msg = None
  img_lock = threading.Lock()

  img_has_subs = False


  alert_boxes = []
  alert_boxes_lock = threading.Lock()


  alerts_dict = []
  alerts_dict_lock = threading.Lock()
  
  last_app_enabled = False
  last_trigger_time = None
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
    self.last_trigger_time = nepi_ros.get_rostime()


    self.initParamServerValues(do_updates = False)
    self.resetParamServer(do_updates = False)
   


    #self.detection_image_pub = rospy.Publisher("~detection_image",Image,queue_size=1)
    # Setup Node Publishers
    self.status_pub = rospy.Publisher("~status", AiAlertsStatus, queue_size=1, latch=True)
    self.alerts_pub = rospy.Publisher("~alerts", AiAlerts, queue_size=1, latch=True)
    self.alert_state_pub = rospy.Publisher("~alert_state", Bool, queue_size=1, latch=True)
    self.alert_trigger_pub = rospy.Publisher("~alert_trigger",Empty,queue_size=1)
    self.image_pub = rospy.Publisher("~alert_image",Image,queue_size=1, latch = True)
    self.snapshot_pub = rospy.Publisher("~snapshot_trigger",Empty,queue_size=1, latch = False)
    self.snapshot_nav_pub = rospy.Publisher(self.base_namespace + "nav_pose_mgr",Empty,queue_size=1, latch = False)
    self.event_pub = rospy.Publisher(self.base_namespace + "event_trigger",Empty,queue_size=1, latch = False)

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
        factory_data_rates[d] = [1.0, 0.0, 100.0] # Default to 1Hz save rate, set last save = 0.0, max rate = 100.0Hz
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
    rospy.Subscriber("~set_alert_delay", Float32, self.setAlertDelayCb, queue_size = 10)
    rospy.Subscriber("~set_clear_delay", Float32, self.setClearDelayCb, queue_size = 10)
    rospy.Subscriber('~set_location_str', String, self.setLocationCb, queue_size = 10)

    rospy.Subscriber("~set_trigger_delay", Float32, self.setSnapshotDelayCb, queue_size = 10)
    rospy.Subscriber('~enable_event_trigger', Bool, self.setEventEnableCb, queue_size = 10)
    rospy.Subscriber('~enable_snapshot_trigger', Bool, self.setSnapshotEnableCb, queue_size = 10)


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
    nepi_ros.timer(nepi_ros.duration(self.UDATE_PROCESS_DELAY), self.updaterCb)
    nepi_ros.timer(nepi_ros.duration(self.IMG_PUB_PROCESS_DELAY), self.imagePubCb)

    time.sleep(1)


    ## Initiation Complete
    nepi_msg.publishMsgInfo(self," Initialization Complete")
    self.publish_status()
    self.alert_state_pub.publish(False)

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
    nepi_ros.set_param(self,'~alert_delay', self.FACTORY_ALERT_DELAY)
    nepi_ros.set_param(self,'~clear_delay', self.FACTORY_CLEAR_DELAY)
    nepi_ros.set_param(self,'~location', "")

    nepi_ros.set_param(self,'~trigger_delay', self.FACTORY_TRIGGER_DELAY)
    nepi_ros.set_param(self,'~snapshot_trigger_enabled', False)
    nepi_ros.set_param(self,'~event_trigger_enabled', False)

    self.last_image_topic = ""

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
      
      self.init_alert_delay = nepi_ros.get_param(self,'~alert_delay', self.FACTORY_ALERT_DELAY)
      self.init_clear_delay = nepi_ros.get_param(self,'~clear_delay', self.FACTORY_CLEAR_DELAY)
      self.init_location = nepi_ros.get_param(self,'~location', "")

      self.init_trigger_delay = nepi_ros.get_param(self,'~trigger_delay', self.FACTORY_TRIGGER_DELAY)
      self.init_snapshot_trigger_enabled = nepi_ros.get_param(self,'~snapshot_trigger_enabled', False)
      self.init_event_trigger_enabled = nepi_ros.get_param(self,'~event_trigger_enabled', False)

      self.resetParamServer(do_updates)




  def resetParamServer(self,do_updates = True):
      nepi_ros.set_param(self,'~app_enabled',self.init_app_enabled)
      nepi_ros.set_param(self,'~last_classiier', self.init_last_classifier)
      nepi_ros.set_param(self,'~selected_classes', self.init_selected_classes)
      nepi_ros.set_param(self,'~alert_delay', self.init_alert_delay)
      nepi_ros.set_param(self,'~clear_delay', self.init_clear_delay)
      nepi_ros.set_param(self,'~location', self.init_location)

      nepi_ros.set_param(self,'~trigger_delay',self.init_trigger_delay)
      nepi_ros.set_param(self,'~snapshot_trigger_enabled', self.init_snapshot_trigger_enabled)
      nepi_ros.set_param(self,'~event_trigger_enabled', self.init_event_trigger_enabled)

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
    status_msg.selected_classes_list = sel_classes
    status_msg.alert_delay_sec = nepi_ros.get_param(self,'~alert_delay', self.init_alert_delay)
    status_msg.clear_delay_sec = nepi_ros.get_param(self,'~clear_delay', self.init_clear_delay)

    status_msg.trigger_delay_sec = nepi_ros.get_param(self,'~trigger_delay', self.init_trigger_delay)
    status_msg.snapshot_trigger_enabled = nepi_ros.get_param(self,'~snapshot_trigger_enabled', self.init_snapshot_trigger_enabled)
    status_msg.event_trigger_enabled = nepi_ros.get_param(self,'~event_trigger_enabled', self.init_event_trigger_enabled)
    self.status_pub.publish(status_msg)

 
  ## Status Publisher
  def publish_alerts(self,active_alert_boxes):
    if self.active_alert == True:
      alerts_msg = AiAlerts()
      stamp = nepi_ros.time_now()
      alerts_msg.header.stamp = stamp
      alerts_msg.date_time_str = nepi_ros.get_datetime_str_from_stamp(stamp)
      alerts_msg.location_str = nepi_ros.get_param(self,'~location',self.init_location)
      alerts_msg.alert_classes_list = active_alert_boxes
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
      if self.image_sub is not None:
        nepi_msg.publishMsgWarn(self," App Disabled, Unsubscribing from Image topic : " + self.last_image_topic)
        self.image_sub.unregister()
        time.sleep(1)
        self.image_sub = None
    elif self.last_app_enabled != app_enabled:
      update_status = True
    self.last_app_enabled = app_enabled


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
        update_status = True
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
      nepi_ros.set_param(self,'~last_classiier', self.current_classifier)
      #nepi_msg.publishMsgWarn(self," Got image topics last and current: " + self.last_image_topic + " " + self.current_image_topic)

      # Update Image Topic Subscriber
      if self.classifier_running == False:
        app_msg += ", Classifier not running"
        self.alerts_dict = dict()
      else:
        app_msg += ", Classifier running"
        if (self.last_image_topic != self.current_image_topic) or (self.image_sub == None and self.current_image_topic != "None") or self.reset_image_topic == True:
          update_status = True
          self.reset_image_topic = False
          image_topic = nepi_ros.find_topic(self.current_image_topic)
          if image_topic == "":
            nepi_msg.publishMsgWarn(self," Could not find image update topic: " + self.current_image_topic)
          elif app_enabled == True and image_topic != "None":
            nepi_msg.publishMsgInfo(self," Found detect Image update topic : " + image_topic)
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
    # Check for img subscribers
    if self.image_sub is not None:
      self.img_has_subs = (self.image_sub.get_num_connections() > 0)

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
    self.publish_status()

  def setAlertDelayCb(self,msg):
    nepi_msg.publishMsgInfo(self,msg)
    val = msg.data
    if val >= 0 and val <= 1:
      nepi_ros.set_param(self,'~alert_delay',val)
    self.publish_status()

  def setClearDelayCb(self,msg):
    nepi_msg.publishMsgInfo(self,msg)
    val = msg.data
    if val >= 0 and val <= 1:
      nepi_ros.set_param(self,'~clear_delay',val)
    self.publish_status()

  def setLocationCb(self,msg):
    ##nepi_msg.publishMsgInfo(self,msg)
    location_str = msg.data
    nepi_ros.set_param(self,'~location', location_str)
    self.publish_status()


  def setSnapshotDelayCb(self,msg):
    #nepi_msg.publishMsgInfo(self,msg)
    val = msg.data
    if val > 0 :
      nepi_ros.set_param(self,'~trigger_delay',val)
    self.publish_status()

        
  def setSnapshotEnableCb(self,msg):
    #nepi_msg.publishMsgInfo(self,msg)
    val = msg.data
    nepi_ros.set_param(self,'~snapshot_trigger_enabled',val)
    self.publish_status()

        
  def setEventEnableCb(self,msg):
    #nepi_msg.publishMsgInfo(self,msg)
    val = msg.data
    nepi_ros.set_param(self,'~event_trigger_enabled',val)
    self.publish_status()
  

  #######################
  ### AI Magnager Callbacks


  ### If object(s) detected, save bounding box info to global
  def objectDetectedCb(self,bounding_boxes_msg):
    app_enabled = nepi_ros.get_param(self,"~app_enabled", self.init_app_enabled)
    ros_timestamp = bounding_boxes_msg.header.stamp

    if app_enabled == False:
      alert_boxes_acquire = False
      self.alert_boxes = []
      alert_boxes_lock = threading.Lock()
      self.alert_classes = []
    else:
      alert_boxes = []
      sel_classes = nepi_ros.get_param(self,'~selected_classes', self.init_selected_classes)
      for box in bounding_boxes_msg.bounding_boxes:
        if box.Class in sel_classes:
          alert_boxes.append(box)
      if len(alert_boxes) > 0:
        # create save dict
        self.alerts_dict_lock.acquire()
        alerts_dict = self.alerts_dict     
        self.alerts_dict_lock.release()

        for box in alert_boxes:
          box_class = box.Class
          if box_class not in alerts_dict.keys():
            alerts_dict[box_class] = dict()
            alerts_dict[box_class]['first_alert_time'] = ros_timestamp
            alerts_dict[box_class]['last_alert_time'] = ros_timestamp
          else:
            if 'first_alert_time' not in alerts_dict[box_class].keys():
              alerts_dict[box_class]['first_alert_time'] = ros_timestamp
            alerts_dict[box_class]['last_alert_time'] = ros_timestamp
            
        self.alert_boxes_lock.acquire()
        self.alert_boxes = alert_boxes      
        self.alert_boxes_lock.release()

        self.alerts_dict_lock.acquire()
        alerts_dict = self.alerts_dict     
        self.alerts_dict_lock.release()

      else:
        self.alert_boxes_lock.acquire()
        self.alert_boxes = []     
        self.alert_boxes_lock.release()



  def imagePubCb(self,timer):
    data_product = 'alert_image'
    has_subscribers = self.img_has_subs
    #nepi_msg.publishMsgWarn(self,"Checking for subscribers: " + str(has_subscribers))
    saving_is_enabled = self.save_data_if.data_product_saving_enabled(data_product)
    snapshot_enabled = self.save_data_if.data_product_snapshot_enabled(data_product)
    should_save = (saving_is_enabled and self.save_data_if.data_product_should_save(data_product)) or snapshot_enabled
    #nepi_msg.publishMsgWarn(self,"Checking for save_: " + str(should_save))
    app_enabled = nepi_ros.get_param(self,"~app_enabled", self.init_app_enabled)
    if app_enabled == False:
      #nepi_msg.publishMsgWarn(self,"Publishing Not Enabled image")
      if not nepi_ros.is_shutdown() and has_subscribers:
        self.app_ne_img.header.stamp = nepi_ros.time_now()
        self.image_pub.publish(self.app_ne_img)
    elif self.image_sub == None:
      if not nepi_ros.is_shutdown() and has_subscribers:
        self.classifier_nr_img.header.stamp = nepi_ros.time_now()
        self.image_pub.publish(self.classifier_nr_img)
    elif has_subscribers or should_save:
      self.img_lock.acquire()
      img_msg = copy.deepcopy(self.img_msg)
      self.img_msg = None
      self.img_lock.release()
      if img_msg is not None:

        self.alert_boxes_lock.acquire()
        alert_boxes = self.alert_boxes   
        self.alert_boxes_lock.release()

        if len(alert_boxes) == 0:
          if img_msg is not None and not nepi_ros.is_shutdown():
            self.image_pub.publish(img_msg)
        else:
          if img_msg is not None:
            current_image_header = img_msg.header
            ros_timestamp = img_msg.header.stamp     
            cv2_img = nepi_img.rosimg_to_cv2img(img_msg).astype(np.uint8)

            for box in alert_boxes:
              #nepi_msg.publishMsgWarn(self," Box: " + str(box))
              class_name = box.Class
              [xmin,xmax,ymin,ymax] = [box.xmin,box.xmax,box.ymin,box.ymax]
              start_point = (xmin, ymin)
              end_point = (xmax, ymax)
              class_name = class_name
              class_color = (0,0,255)
              line_thickness = 2
              cv2.rectangle(cv2_img, start_point, end_point, class_color, thickness=line_thickness)

            # Publish new image to ros
            if not nepi_ros.is_shutdown() and has_subscribers: #and has_subscribers:
                #Convert OpenCV image to ROS image
                cv2_shape = cv2_img.shape
                if  cv2_shape[2] == 3:
                  encode = 'bgr8'
                else:
                  encode = 'mono8'
                img_out_msg = nepi_img.cv2img_to_rosimg(cv2_img, encoding=encode)
                img_out_msg.header.stamp = ros_timestamp
                self.image_pub.publish(img_out_msg)
            # Save Data if \
            if should_save:
              nepi_save.save_img2file(self,data_product,cv2_img,ros_timestamp,save_check = False)

  def imageCb(self,image_msg):   
      #nepi_msg.publishMsgWarn(self,"Got image msg: ") 
      self.img_lock.acquire()
      self.img_msg = copy.deepcopy(self.last_img_msg)
      self.img_lock.release()
      self.last_img_msg = copy.deepcopy(image_msg)



  ### Monitor Output of AI model to clear detection status
  def foundObjectCb(self,found_obj_msg):
    app_enabled = nepi_ros.get_param(self,"~app_enabled", self.init_app_enabled)
    ros_timestamp = found_obj_msg.header.stamp
    #Clean Up Detection and Alert data
    if found_obj_msg.count == 0:
      self.alert_boxes_lock.acquire()
      self.alert_boxes = []     
      self.alert_boxes_lock.release()

    # Update alert dict and status info
    self.alerts_dict_lock.acquire()
    alerts_dict = self.alerts_dict     
    self.alerts_dict_lock.release()

    # Purge old alerts
    clear_delay = nepi_ros.get_param(self,'~clear_delay', self.init_clear_delay)
    purge_alert_list = []
    for key in alerts_dict.keys():
      last_alert_time =(ros_timestamp.to_sec() - alerts_dict[key]['last_alert_time'].to_sec())
      if last_alert_time > clear_delay:
        purge_alert_list.append(key)
    for alert in purge_alert_list:
      del alerts_dict[alert]
    self.alerts_dict_lock.acquire()
    self.alerts_dict = alerts_dict
    self.alerts_dict_lock.release()

    # Check current alert trigger time
    active_alert = False
    active_alert_classes = []
    alert_delay = nepi_ros.get_param(self,'~alert_delay', self.init_alert_delay)
    for key in alerts_dict.keys():
      first_alert_time =(ros_timestamp.to_sec() - alerts_dict[key]['first_alert_time'].to_sec())
      if first_alert_time > alert_delay:
        active_alert = True
        if key not in active_alert_classes:
          active_alert_classes.append(key)
    self.alert_classes = active_alert_classes
    self.active_alert = active_alert
    self.alert_state_pub.publish(self.active_alert)
    if len(active_alert_classes) > 0:
      self.publish_alerts(active_alert_classes)
      alerts_save_dict = dict()
      alerts_save_dict['timestamp'] = nepi_ros.get_datetime_str_from_stamp(ros_timestamp)
      alerts_save_dict['location'] = nepi_ros.get_param(self,'~location',self.init_location)
      alerts_save_dict['alert_classes_list'] = active_alert_classes
      nepi_save.save_dict2file(self,'alert_data',alerts_save_dict,ros_timestamp,save_check = True)


    # Do stuff on active alert state
    if active_alert == True:
        trigger_delay = nepi_ros.get_param(self,'~trigger_delay', self.init_trigger_delay)
        trigger_time = (ros_timestamp.to_sec() - self.last_trigger_time.to_sec())
        if (trigger_time > trigger_delay):
          self.last_trigger_time = ros_timestamp
          self.alert_trigger_pub.publish(Empty())
          snapshot_trigger_enabled = nepi_ros.get_param(self,'~snapshot_trigger_enabled', self.init_snapshot_trigger_enabled)
          if snapshot_trigger_enabled:
            self.snapshot_pub.publish(Empty())
            self.snapshot_nav_pub.publish(Empty())
          event_trigger_enabled = nepi_ros.get_param(self,'~event_trigger_enabled', self.init_event_trigger_enabled)
          if event_trigger_enabled:
            self.event_pub.publish(Empty())
        # Publish and save active alert boxes
        self.alert_boxes_lock.acquire()
        alert_boxes = self.alert_boxes    
        self.alert_boxes_lock.release()

    else:
      self.last_trigger_time = ros_timestamp

  
               
    
  #######################
  # Node Cleanup Function
  
  def cleanup_actions(self):
    nepi_msg.publishMsgInfo(self," Shutting down: Executing script cleanup actions")


#########################################
# Main
#########################################
if __name__ == '__main__':
  NepiAiAlertsApp()







