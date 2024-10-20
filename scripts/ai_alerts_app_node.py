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
from nepi_ros_interfaces.msg import BoundingBox, BoundingBoxes

from nepi_ros_interfaces.srv import ImageClassifierStatusQuery, ImageClassifierStatusQueryRequest

from nepi_app_ai_alerts.msg import AiAlertsStatus, AiAlerts

from nepi_edge_sdk_base.save_data_if import SaveDataIF
from nepi_edge_sdk_base.save_cfg_if import SaveCfgIF

# Do this at the end
#from scipy.signal import find_peaks

#########################################
# Node Class
#########################################

class NepiAiTargetingApp(object):
  AI_MANAGER_NODE_NAME = "ai_detector_mgr"

  #Set Initial Values
  
  FACTORY_SENSITIVITY = 0.5

  NONE_CLASSES_DICT = dict()

  data_products = ["alert_image","alert_bounding_boxes"]
  output_image_options = ["None","Alert_Image"]
  
  current_classifier = "None"
  current_classifier_state = "None"
  classes_list = []
  current_classifier_classes = "[]"

  current_image_topic = "None"
  current_image_header = Header()
  image_source_topic = ""
  img_width = 0
  img_height = 0
  image_sub = None
  last_img_msg = None

  bbs_msg = None

  last_image_topic = "None"
  
  selected_classes = dict()
 
  alerts_list = []
  alerts_dict = dict()
  active_alert = False

  status_pub = None
  alert_trigger_pub = None
  alerts_pub = None
  alerts_image_pub = None
  alerts_boxes_2d_pub = None

  seq = 0

  #has_subscribers_detect_img = False
  has_subscribers_alert_img = False

  classifier_running = False
  classifier_loading_progress = 0.0
  classifier_threshold = 0.3

  sensitivity_count = 10
  no_object_count = 0
  #######################
  ### Node Initialization
  DEFAULT_NODE_NAME = "ai_alerts_app" # Can be overwitten by luanch command
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
   
    # Message Image to publish when detector not running
    message = "WAITING FOR AI DETECTOR TO START"
    cv2_img = nepi_img.create_message_image(message)
    self.ros_message_img = nepi_img.cv2img_to_rosimg(cv2_img)

    #self.detection_alerts_image_pub = rospy.Publisher("~detection_image",Image,queue_size=1)
    # Setup Node Publishers
    self.status_pub = rospy.Publisher("~status", AiAlertsStatus, queue_size=1, latch=True)
    self.alerts_pub = rospy.Publisher("~alerts", AiAlert, queue_size=1, latch=True)
    self.alerts_boxes_2d_pub = rospy.Publisher("~alert_boxes_2d", BoundingBoxes, queue_size=1)
    self.alert_trigger_pub = rospy.Publisher("~alert_trigger",Empty,queue_size=1)
    self.alerts_image_pub = rospy.Publisher("~alert_image",Image,queue_size=1, latch = True)
    time.sleep(1)
    self.ros_message_img.header.stamp = nepi_ros.time_now()
    self.alerts_image_pub.publish(self.ros_message_img)

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
    set_image_input_sub = rospy.Subscriber('~use_live_image', Bool, self.setImageLiveCb, queue_size = 10)
    add_all_sub = rospy.Subscriber('~add_all_alert_classes', Empty, self.addAllClassesCb, queue_size = 10)
    remove_all_sub = rospy.Subscriber('~remove_all_alert_classes', Empty, self.removeAllClassesCb, queue_size = 10)
    add_class_sub = rospy.Subscriber('~add_alert_class', String, self.addClassCb, queue_size = 10)
    remove_class_sub = rospy.Subscriber('~remove_alert_class', String, self.removeClassCb, queue_size = 10)
    age_filter_sub = rospy.Subscriber("~set_alert_sensitivity", Float32, self.setSensitivity, queue_size = 10)
    set_loc_sub = rospy.Subscriber('~set_location', String, self.setLocationCb, queue_size = 10)

    # Start an AI manager status monitoring thread
    AI_MGR_STATUS_SERVICE_NAME = self.ai_mgr_namespace  + "/img_classifier_status_query"
    self.get_ai_mgr_status_service = rospy.ServiceProxy(AI_MGR_STATUS_SERVICE_NAME, ImageClassifierStatusQuery)
    time.sleep(1)
    nepi_ros.timer(nepi_ros.duration(1), self.updaterCb)

    # Start AI Manager Subscribers
    FOUND_OBJECT_TOPIC = self.ai_mgr_namespace  + "/found_object"
    rospy.Subscriber(FOUND_OBJECT_TOPIC, ObjectCount, self.found_object_callback, queue_size = 1)
    BOUNDING_BOXES_TOPIC = self.ai_mgr_namespace  + "/bounding_boxes"
    rospy.Subscriber(BOUNDING_BOXES_TOPIC, BoundingBoxes, self.object_detected_callback, queue_size = 1)

    nepi_ros.timer(nepi_ros.duration(1), self.updateHasSubscribersThread)

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
    nepi_ros.set_param(self,'~last_classifier', "")
    nepi_ros.set_param(self,'~use_live_image',True)
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
      self.init_last_classifier = nepi_ros.get_param(self,"~last_classifier", "")
      self.init_use_live_image = nepi_ros.get_param(self,'~use_live_image',True)
      self.init_selected_classes = nepi_ros.get_param(self,'~selected_classes', [])
      self.init_sensitivity = nepi_ros.get_param(self,'~sensitivity', self.FACTORY_SENSITIVITY)
      self.init_snapshot_enabled = nepi_ros.get_param(self,'~snapshot_enabled', False)
      self.init_snapshot_delay = nepi_ros.get_param(self,'~snapshot_delay', 5)
      self.init_location = nepi_ros.get_param(self,'~location', "")
      self.resetParamServer(do_updates)

  def resetParamServer(self,do_updates = True):
      nepi_ros.set_param(self,'~last_classiier', self.init_last_classifier)
      nepi_ros.get_param(self,'~use_live_image',self.init_use_live_image)
      nepi_ros.set_param(self,'~selected_classes', self.init_selected_classes)
      nepi_ros.set_param(self,'~sensitivity', self.init_sensitivity)
      nepi_ros.set_param(self,'~snapshot_enabled', self.init_snapshot_enabled)
      nepi_ros.set_param(self,'~snapshot_delay',self.init_snapshot_delay)
      nepi_ros.get_param(self,'~location', self.init_location)
      if do_updates:
          self.updateFromParamServer()
          self.publish_status()


  ###################
  ## Status Publisher
  def publish_status(self):
    status_msg = AiAlertsStatus()

    status_msg.location = nepi_ros.get_param(self,'~location',self.init_location)
    status_msg.classifier_running = self.classifier_running

    status_msg.classifier_name = self.current_classifier
    status_msg.classifier_state = self.current_classifier_state
    status_msg.use_live_image = nepi_ros.get_param(self,'~use_live_image',self.init_use_live_image)

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
    if len(sel_classes_list) == 0:
      sel_classes_list = ['None']
    status_msg.selected_classes_list = (sel_classes)
    nepi_ros.set_param(self,'~selected_classes', sel_classes)

    status_msg.sensitivity = nepi_ros.get_param(self,'~sensitivity', self.init_sensitivity)
    status_msg.snapshot_enabled = nepi_ros.get_param(self,'~snapshot_enabled', self.init_snapshot_enabled)
    status_msg.snapshot_delay = nepi_ros.get_param(self,'~snapshot_delay', self.init_snapshot_delay)

    status_msg.active_alert = self.active_alert

    self.status_pub.publish(status_msg)

 
  ## Status Publisher
  def publish_alerts(self):
    alerts_msg = AiAlerts()
    alerts_msg.header.stamp = nepi_ros.time_now()
    alerts_msg.location = nepi_ros.get_param(self,'~location',self.init_location)
    alerts_msg.alert_classes_list = self.alerts_list
    self.alerts_pub.publish(alerts_msg)     
    
 

  def updaterCb(self,timer):
    try:
      ai_mgr_status_response = self.get_ai_mgr_status_service()
      #nepi_msg.publishMsgInfo(self," Got classifier status  " + str(ai_mgr_status_response))
    except Exception as e:
      nepi_msg.publishMsgWarn(self,"Failed to call AI MGR STATUS service" + str(e))
      return
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
    if self.classifier_running:
      use_live_image = nepi_ros.get_param(self,'~use_live_image',self.init_use_live_image)
      if (self.last_image_topic != self.current_image_topic) or (self.image_sub == None and self.current_image_topic != "None"):
        image_topic = ""
        if use_live_image:
          image_topic = nepi_ros.find_topic(self.current_image_topic)
        if image_topic == "":
          source_topic = AI_MGR_STATUS_SERVICE_NAME = self.ai_mgr_namespace  + "/source_image"
          image_topic = nepi_ros.find_topic(source_topic)
        nepi_msg.publishMsgInfo(self," Got detect image update topic update : " + image_topic)
        update_status = True
        if image_topic != "":
          self.alerts_running = True
          update_status = True
          if self.image_sub != None:
            nepi_msg.publishMsgWarn(self," Unsubscribing to Image topic : " + image_topic)
            self.image_sub.unregister()
            self.image_sub = None
          time.sleep(1)
          if self.alerts_image_pub is None:
            #nepi_msg.publishMsgWarn(self," Creating Image publisher ")
            self.alerts_image_pub = rospy.Publisher("~image",Image,queue_size=1)
            time.sleep(1)
          nepi_msg.publishMsgInfo(self," Subscribing to Image topic : " + image_topic)

          self.image_sub = rospy.Subscriber(image_topic, Image, self.alertsImageCb, queue_size = 1)
    elif self.classifier_running == False or self.current_image_topic == "None" or self.current_image_topic == "":  # Turn off alerts subscribers and reset last image topic
      self.alerts_running = False
      self.alerts_dict = dict()
      if self.image_sub != None:
        nepi_msg.publishMsgWarn(self," Unsubscribing to Image topic : " + self.current_image_topic)
        self.image_sub.unregister()
        self.image_sub = None
      update_status = True
      time.sleep(1)
    # Publish warning image if not running
    if self.classifier_running == False or self.image_sub == None:
      self.ros_message_img.header.stamp = nepi_ros.time_now()
      self.alerts_image_pub.publish(self.ros_message_img)
    # Save last image topic for next check
    self.last_image_topic = self.current_image_topic

    # Do alert check
    sensitivity = nepi_ros.get_param(self,"~sensitivity",self.init_sensitivity)
    sensitivity_count = int(10 * sensitivity) 
    alerts_list = []
    for class_name in self.alerts_dict():
      count = self.alerts_dict[class_name]
      if count > sensitivity_count and class_name not in alerts_list:
        alerts_list.append(class_name)
    self.alerts_list = alerts_list
    if len(alerts_list) > 0: 
      self.alert_trigger_pub.publish(Empty)
      self.active_alert = True
      self.publish_alerts()
    else:
      self.active_alert = False
    if update_status == True:
      self.publish_status()




  ###################
  ## AI App Callbacks

  def setImageLiveCb(self,msg):
    ##nepi_msg.publishMsgInfo(self,msg)
    live = msg.data
    current_live = nepi_ros.get_param(self,'~use_live_image',self.init_use_live_image)
    if live != current_live:
      self.last_image_topic = None # Will force resubscribe later
      nepi_ros.set_param(self,'~use_live_image',live)
    self.publish_status()

  def addAllClassesCb(self,msg):
    ##nepi_msg.publishMsgInfo(self,msg)
    classes = self.classes_list
    depth = nepi_ros.get_param(self,'~default_alert_depth',self.init_default_alert_depth)
    selected_dict = dict()
    for Class in classes:
      selected_dict[Class] = {'depth': depth }
    nepi_ros.set_param(self,'~selected_classes', selected_dict)
    self.publish_status()

  def removeAllClassesCb(self,msg):
    ##nepi_msg.publishMsgInfo(self,msg)
    nepi_ros.set_param(self,'~selected_classes', dict())
    self.publish_status()

  def addClassCb(self,msg):
    ##nepi_msg.publishMsgInfo(self,msg)
    class_name = msg.data
    class_depth_m = nepi_ros.get_param(self,'~default_alert_depth',  self.init_default_alert_depth)
    if class_name in self.classes_list:
      selected_classes = nepi_ros.get_param(self,'~selected_classes', self.init_selected_classes)
      selected_classes[class_name] = {'depth': class_depth_m}
      nepi_ros.set_param(self,'~selected_classes', selected_classes)
    self.publish_status()

  def removeClassCb(self,msg):
    ##nepi_msg.publishMsgInfo(self,msg)
    class_name = msg.data
    selected_classes = nepi_ros.get_param(self,'~selected_classes', self.init_selected_classes)
    if class_name in selected_classes.keys():
      del selected_classes[class_name]
      nepi_ros.set_param(self,'~selected_classes', selected_classes)
    self.publish_status()

  def setSensitivity(self,msg):
    #nepi_msg.publishMsgInfo(self,msg)
    val = msg.data
    if val >= 0 and val <= 1:
      nepi_ros.set_param(self,'~sensitivity',val)
    self.publish_status()

  def setSnapshotEnable(self,msg):
    #nepi_msg.publishMsgInfo(self,msg)
    val = msg.data
    nepi_ros.set_param(self,'~snapshot_enabled',val)
    self.publish_status()

  def setSnapshotDelay(self,msg):
    #nepi_msg.publishMsgInfo(self,msg)
    val = msg.data
    if val > 0 :
      nepi_ros.set_param(self,'~snapshot_delay',val)
    self.publish_status()

  def updateHasSubscribersThread(self,timer):
    #self.has_subscribers_detect_img = (self.detection_image_pub.get_num_connections() > 0)
    if self.alerts_image_pub is not None:
      self.has_subscribers_alert_img = (self.alerts_image_pub.get_num_connections() > 0)
    else:
      self.has_subscribers_alert_img = False
  

  #######################
  ### AI Magnager Callbacks



  ### Monitor Output of AI model to clear detection status
  def found_object_callback(self,found_obj_msg):
    # Must reset alert lists if no alerts are detected
    if found_obj_msg.count == 0:
      #print("No objects detected")
      self.bbs_msg = None
      self.no_object_count += 1
    else:
      self.no_object_count = 0
    if self.no_object_count > self.sensitivity_count:
      self.alerts_list = []
      self.alerts_dict = dict()
      self.active_alert = False



  ### If object(s) detected, save bounding box info to global
  def object_detected_callback(self,bounding_boxes_msg):
    self.bbs_msg=bounding_boxes_msg
    # Check for alert class
    box_classes = []
    for box in bounding_boxes_msg.bounding_boxes:
      if box.Class not in box_classes:
        box_classes.append(box.Class)
    for box_class in box_classes:
      if box_class in self.alerts_dict.keys():
        self.alerts_dict[box.Class] += 1
      else:
         self.alerts_dict[box.Class] = 1
    for alert_class in self.alerts_dict.keys():
      if alert_class not in box_classes:
        self.alerts_dict[box.Class] -= 1
        if self.alerts_dict[box.Class] < 0:
          self.alerts_dict[box.Class] = 0

  def alertsImageCb(self,img_in_msg):    
    data_product = 'alert_image'
    output_image = nepi_ros.get_param(self,'~selected_output_image', self.init_selected_output_image)
    if self.alerts_image_pub is not None:
        has_subscribers =  self.has_subscribers_alert_img
        saving_is_enabled = self.save_data_if.data_product_saving_enabled(data_product)
        data_should_save  = self.save_data_if.data_product_should_save(data_product) and saving_is_enabled
        snapshot_enabled = self.save_data_if.data_product_snapshot_enabled(data_product)
        save_data = (saving_is_enabled and data_should_save) or snapshot_enabled
        self.current_image_header = img_in_msg.header
        ros_timestamp = img_in_msg.header.stamp     
        self.img_height = img_in_msg.height
        self.img_width = img_in_msg.width
        cv2_in_img = nepi_img.rosimg_to_cv2img(img_in_msg)
        cv2_img = copy.deepcopy(cv2_in_img)
        alert_list = copy.deepcopy(self.alert_list)
        bbs_msg = copy.deepcopy(self.bbs_msg)

        # Process Alerts Image if Needed
        if alert_list == None:
            alert_list = []
        if len(alert_list) > 0:
            for box in bbs_msg.bounding_boxes:
              class_name = box['Class']
              if class_name in alert_list:
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
            img_out_msg = nepi_img.cv2img_to_rosimg(cv2_img, encoding='bgr8')
            self.alerts_image_pub.publish(img_out_msg)
        # Save Data if Time
        if save_data:
          nepi_save.save_img2file(self,data_product,cv2_img,ros_timestamp,save_check = False)
               
    
  #######################
  # Node Cleanup Function
  
  def cleanup_actions(self):
    nepi_msg.publishMsgInfo(self," Shutting down: Executing script cleanup actions")


#########################################
# Main
#########################################
if __name__ == '__main__':
  NepiAiTargetingApp()







