/*
 * Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
 *
 * This file is part of nepi-engine
 * (see https://github.com/nepi-engine).
 *
 * License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
 */
import React, { Component } from "react"
import { observer, inject } from "mobx-react"

import Section from "./Section"
//import EnableAdjustment from "./EnableAdjustment"
import Button, { ButtonMenu } from "./Button"
import {SliderAdjustment} from "./AdjustmentWidgets"
import Label from "./Label"
import { Column, Columns } from "./Columns"
import Input from "./Input"
import Select, { Option } from "./Select"
import Styles from "./Styles"
import Toggle from "react-toggle"


import AiDetectorMgr from "./NepiMgrAiDetector"
import CameraViewer from "./CameraViewer"
import NepiIFSaveData from "./Nepi_IF_SaveData"


import {round, convertStrToStrList, createMenuListFromStrList, onDropdownSelectedSendStr, onUpdateSetStateValue, onEnterSendFloatValue, onEnterSendIntValue, onEnterSetStateFloatValue} from "./Utilities"

@inject("ros")
@observer

// Component that contains the  Pointcloud App Viewer Controls
class AppAiAlerts extends Component {
  constructor(props) {
    super(props)

    // these states track the values through  Status messages
    this.state = {

      appName: "app_ai_alerts",
      appNamespace: null,

      location_str: "",
      sensitivity: 0.5,
      snapshot_enabled: false,
      snapshot_delay: 5,

      classifier_running: false,
      classifier_name: null,
      classifier_state: null,

      use_live_image: true,
      use_last_image: true,
      image_topic: null,
      
      available_classes_list: [],
      last_classes_list: [],
      selected_classes_list:[],
        
      viewableTopics: false,

      statusListener: null,

      connected: false,
      needs_update: true

    }
  
    this.getAppNamespace = this.getAppNamespace.bind(this)
    this.statusListener = this.statusListener.bind(this)
    this.updateStatusListener = this.updateStatusListener.bind(this)
    
    this.onToggleClassSelection = this.onToggleClassSelection.bind(this)
    this.getClassOptions = this.getClassOptions.bind(this)
    this.toggleViewableTopics = this.toggleViewableTopics.bind(this)

  }

  getAppNamespace(){
    const { namespacePrefix, deviceId} = this.props.ros
    var appNamespace = null
    if (namespacePrefix !== null && deviceId !== null){
      appNamespace = "/" + namespacePrefix + "/" + deviceId + "/" + this.state.appName
    }
    return appNamespace
  }

  // Callback for handling ROS Status messages
  statusListener(message) {
    this.setState({

    location_str: message.location_str,
    sensitivity: message.sensitiivty,
    snapshot_enabled: message.snapshot_enabled,
    snapshot_delay: message.snapshot_delay_sec,

    classifier_running: message.classifier_running,

    classifier_name: message.classifier_name,
    classifier_state: message.classifier_state,
    use_live_image: message.use_live_image,
    use_last_image: message.use_last_image,
    image_topic: message.image_topic,
    
    available_classes_list: message.available_classes_list,
    selected_classes_list: message.selected_classes_list,
    selected_classes_depth_list: message.selected_classes_depth_list
    })

    this.setState({
      connected: true
    })

    const last_classes_list = this.state.last_classes_list
    this.setState({
      last_classes_list: this.state.available_classes_list
    })
    if (last_classes_list !== this.state.available_classes_list){
      this.render()
    }
  }

    // Function for configuring and subscribing to Status
    updateStatusListener() {
      const appNamespace = this.getAppNamespace()
      const statusNamespace = appNamespace + '/status'
      if (this.state.statusListener) {
        this.state.statusListener.unsubscribe()
      }
      var statusListener = this.props.ros.setupStatusListener(
            statusNamespace,
            "nepi_app_ai_alerts/AiAlertsStatus",
            this.statusListener
          )
      this.setState({ 
        statusListener: statusListener,
        needs_update: false
      })
      this.render()
    }


  // Lifecycle method called when compnent updates.
  // Used to track changes in the topic
  componentDidUpdate(prevProps, prevState, snapshot) {
    const namespace = this.getAppNamespace()
    const namespace_updated = (prevState.appNamespace !== namespace && namespace !== null)
    const needs_update = (this.state.needs_update && namespace !== null)
    if (namespace_updated || needs_update) {
      if (namespace.indexOf('null') === -1){
        this.setState({appNamespace: namespace})
        this.updateStatusListener()
      } 
    }
  }


  // Lifecycle method called just before the component umounts.
  // Used to unsubscribe to Status message
  componentWillUnmount() {
    if (this.state.statusListener) {
      this.state.statusListener.unsubscribe()
    }
  }


  // Function for creating image topic options.
  getClassOptions() {
  const classesList = this.state.available_classes_list
  var items = []
  items.push(<Option>{"None"}</Option>)
  items.push(<Option>{"All"}</Option>)
  if (classesList.length > 0 ){
    for (var i = 0; i < classesList.length; i++) {
        if (classesList[i] !== 'None'){
          items.push(<Option value={classesList[i]}>{classesList[i]}</Option>)
        }
    }
  }
  return items
  }


  toggleViewableTopics() {
    const set = !this.state.viewableTopics
    this.setState({viewableTopics: set})
  }


  onToggleClassSelection(event){
    const {sendTriggerMsg, sendStringMsg} = this.props.ros
    const appNamespace = this.getAppNamespace()
    const classSelection = event.target.value
    const selectedClassesList = this.state.selected_classes_list
    const addAllNamespace = appNamespace + "/add_all_alert_classes"
    const removeAllNamespace = appNamespace + "/remove_all_alert_classes"
    const addNamespace = appNamespace + "/add_alert_class"
    const removeNamespace = appNamespace + "/remove_alert_class"
    if (appNamespace){
      if (classSelection === "None"){
          sendTriggerMsg(removeAllNamespace)
      }
      else if (classSelection === "All"){
        sendTriggerMsg(addAllNamespace)
    }
      else if (selectedClassesList.indexOf(classSelection) !== -1){
        sendStringMsg(removeNamespace,classSelection)
      }
      else {
        sendStringMsg(addNamespace,classSelection)
      }
    }
  }


 

  renderApp() {
    const {sendBoolMsg, sendTriggerMsg,} = this.props.ros
    const appNamespace = this.getAppNamespace()
    const classOptions = this.getClassOptions()
    const selectedClasses = this.state.selected_classes_list
    const NoneOption = <Option>None</Option>
    const classifier_running = this.state.classifier_running
    return (
      <Section title={"AI Alerts App"}>

        <Columns>
        <Column>
        

            <Label title="Select Class Filters"> </Label>

                    <div onClick={this.toggleViewableTopics} style={{backgroundColor: Styles.vars.colors.grey0}}>
                      <Select style={{width: "10px"}}/>
                    </div>
                    <div hidden={this.state.viewableTopics === false}>
                    {classOptions.map((Class) =>
                    <div onClick={this.onToggleClassSelection}
                      style={{
                        textAlign: "center",
                        padding: `${Styles.vars.spacing.xs}`,
                        color: Styles.vars.colors.black,
                        backgroundColor: (selectedClasses.includes(Class.props.value))? Styles.vars.colors.blue : Styles.vars.colors.grey0,
                        cursor: "pointer",
                        }}>
                        <body class_name ={Class} style={{color: Styles.vars.colors.black}}>{Class}</body>
                    </div>
                    )}
                    </div>


              </Column>
              <Column>

          <Label title="Use Live Image">
              <Toggle
              checked={this.state.use_live_image===true}
              onClick={() => sendBoolMsg(appNamespace + "/use_live_image",!this.state.use_live_image)}>
              </Toggle>
        </Label>
        
        <Label title="Use Last Image">
              <Toggle
              checked={this.state.use_last_image===true}
              onClick={() => sendBoolMsg(appNamespace + "/use_last_image",!this.state.use_last_image)}>
              </Toggle>
        </Label>

              <ButtonMenu>
            <Button onClick={() => sendTriggerMsg( appNamespace + "/reset_app")}>{"Reset App"}</Button>
          </ButtonMenu>

            <ButtonMenu>
              <Button onClick={() => sendTriggerMsg(appNamespace + "/save_config")}>{"Save Config"}</Button>
        </ButtonMenu>

        <ButtonMenu>
              <Button onClick={() => sendTriggerMsg( appNamespace + "/reset_config")}>{"Reset Config"}</Button>
        </ButtonMenu>


              </Column>
              </Columns>




        <SliderAdjustment
          title={"Alert Sensitivity Ratio"}
          msgType={"std_msgs/float32"}
          adjustment={this.state.sensitivity}
          topic={appNamespace + "/set_sensitivity"}
          scaled={0.01}
          min={0}
          max={100}
          tooltip={""}
          unit={"%"}
      />


      </Section>

    
    )
  }


  renderImageViewer(){
    const connected = this.state.connected
    const namespace = this.getAppNamespace()
    const appNamespace = (connected) ? namespace: null
    const imageNamespace = (connected) ? appNamespace + "/alert_image" : null
    return (

      <CameraViewer
        imageTopic={imageNamespace}
        title={this.state.selected_output_image}
        hideQualitySelector={false}
      />

      )
    }  

  render() {
    const connected = this.state.connected
    const namespace = this.getAppNamespace()
    const appNamespace = (connected) ? namespace: null
    return (

      <Columns>
      <Column equalWidth={false}>

  
      <label style={{fontWeight: 'bold'}} align={"left"} textAlign={"left"}>
          {"Connecting"}
         </label>
      

      {this.renderImageViewer()}


      </Column>
      <Column>


      <AiDetectorMgr
              title={"Nepi_Mgr_AI_Detector"}
          />

      {this.renderApp()}

        <NepiIFSaveData
          saveNamespace={appNamespace}
          title={"Nepi_IF_SaveData"}
        />

      </Column>
      </Columns>

      )
    }  

}
export default AppAiAlerts
