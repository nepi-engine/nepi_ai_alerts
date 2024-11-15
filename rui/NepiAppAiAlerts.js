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
import BooleanIndicator from "./BooleanIndicator"


import AiDetectorMgr from "./NepiMgrAiDetector"
import CameraViewer from "./CameraViewer"
import NepiIFSaveData from "./Nepi_IF_SaveData"


import {round, onChangeSwitchStateValue, onUpdateSetStateValue, onEnterSendFloatValue} from "./Utilities"

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

      app_enabled: false,
      app_msg: "Connecting",
      image_name: "alert_image",
      show_detector_box: false,

      location_str: "",

      classifier_running: false,

      image_topic: null,
      
      available_classes_list: [],
      last_classes_list: [],
      selected_classes_list:[],

      sensitivity: null,
      snapshot_enabled: null,
      snapshot_delay: null,

      active_alert: false,
        
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
      if (this.state.connected === false){
        const pub_status_topic = appNamespace + "/publish_status"
        this.props.ros.sendTriggerMsg(pub_status_topic)
      }
    }
    return appNamespace
  }

  // Callback for handling ROS Status messages
  statusListener(message) {
    this.setState({

    app_enabled: message.app_enabled,
    app_msg: message.app_msg,

    location_str: message.location_str,

    image_topic: message.image_topic,

    classifier_running: message.classifier_running,
    available_classes_list: message.available_classes_list,
    selected_classes_list: message.selected_classes_list,

    sensitivity: message.sensitivity,
    snapshot_enabled: message.snapshot_enabled,
    snapshot_delay: message.snapshot_delay_sec,

    active_alert: message.active_alert

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
    const classOptions = this.getClassOptions()
    const selectedClasses = this.state.selected_classes_list
    const NoneOption = <Option>None</Option>
    const classifier_running = this.state.classifier_running
    const connected = this.state.connected === true
    const appNamespace = this.getAppNamespace()
    const classes_sel = selectedClasses[0] !== "" && selectedClasses[0] !== "None"

    return (
      <Section title={"AI Alerts App"}>

        <Columns>
        <Column>

        <Columns>
        <Column>

        <div hidden={(connected === true)}>

      <pre style={{ height: "40px", overflowY: "auto" ,fontWeight: 'bold' , color: Styles.vars.colors.Green, textAlign: "left" }}>
          {"Loading"}
        </pre>

      </div>

      <div hidden={(connected === false)}>

        <Label title="Enable App">
            <Toggle
            checked={this.state.app_enabled===true}
            onClick={() => sendBoolMsg(appNamespace + "/enable_app",!this.state.app_enabled)}>
            </Toggle>
      </Label>

      </div>


          </Column>
        <Column>


        </Column>
      </Columns>


      <div hidden={(connected !== true || this.state.app_enabled !== true)}>

          <Columns>
          <Column>


      <Label title={"Classifier Running"}>
        <BooleanIndicator value={this.state.classifier_running} />
      </Label>



            </Column>
          <Column>

          <Label title={"Alert Classes Selected"}>
        <BooleanIndicator value={classes_sel} />
      </Label>
  
          </Column>
        </Columns>

   

        <div style={{ borderTop: "1px solid #ffffff", marginTop: Styles.vars.spacing.medium, marginBottom: Styles.vars.spacing.xs }}/>

       
        <label style={{fontWeight: 'bold'}} align={"left"} textAlign={"left"}>
          {"App Settings"}
         </label>


        <Columns>
          <Column>



         <Label title="Select Alert Classes"> </Label>

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

{/*}
        <SliderAdjustment
          title={"Alert Sensitivity"}
          msgType={"std_msgs/float32"}
          adjustment={this.state.sensitivity}
          topic={appNamespace + "/set_sensitivity"}
          scaled={0.01}
          min={0}
          max={100}
          tooltip={""}
          unit={"%"}
      />
*/}

          <Label title={"Location"}>
                <Input
                  value={this.state.location_str}
                  id="Location"
                  onChange= {(event) => onUpdateSetStateValue.bind(this)(event,"location_str")}
                  onKeyDown= {(event) => onEnterSendFloatValue.bind(this)(event,appNamespace + "/set_location_str")}
                  style={{ width: "80%" }}
                />
              </Label>


          <Label title="Take Snapshot on Alerts">
              <Toggle
              checked={this.state.snapshot_enabled===true}
              onClick={() => sendBoolMsg(appNamespace + "/set_snapshot_enable",!this.state.snapshot_enabled)}>
              </Toggle>
        </Label>


        <Label title={"Snapshot delay (sec)"}>
                <Input
                  value={this.state.snapshot_delay}
                  id="Snapshot_Delay"
                  onChange= {(event) => onUpdateSetStateValue.bind(this)(event,"snapshot_delay")}
                  onKeyDown= {(event) => onEnterSendFloatValue.bind(this)(event,appNamespace + "/set_snapshot_delay")}
                  style={{ width: "80%" }}
                />
          </Label>

        </Column>
        </Columns>



      <div style={{ borderTop: "1px solid #ffffff", marginTop: Styles.vars.spacing.medium, marginBottom: Styles.vars.spacing.xs }}/>

      <Columns>
          <Column>

            <ButtonMenu>
              <Button onClick={() => sendTriggerMsg( appNamespace + "/reset_app")}>{"Reset App"}</Button>
            </ButtonMenu>

            </Column>
          <Column>

              <ButtonMenu>
                <Button onClick={() => sendTriggerMsg(appNamespace + "/save_config")}>{"Save Config"}</Button>
          </ButtonMenu>

          </Column>
          <Column>

          <ButtonMenu>
                <Button onClick={() => sendTriggerMsg( appNamespace + "/reset_config")}>{"Reset Config"}</Button>
          </ButtonMenu>

  
          </Column>
        </Columns>

        </div>

      </Column>
        </Columns>

      </Section>

    
    )
  }


  render() {
    const connected = this.state.connected === true
    const appNamespace = (connected) ? this.getAppNamespace() : null
    const show_detector_box = this.state.show_detector_box
    const imageNamespace = appNamespace + '/' + this.state.image_name

    return (

      <Columns>
      <Column equalWidth={true}>

       

      <CameraViewer
        imageTopic={imageNamespace}
        title={this.state.image_name}
        hideQualitySelector={false}
      />


      </Column>
      <Column>


      <Columns>
      <Column>

      <Label title="Show Detector Settings">
              <Toggle
              checked={(this.state.show_detector_box === true)}
              onClick={() => onChangeSwitchStateValue.bind(this)("show_detector_box",this.state.show_detector_box)}>
              </Toggle>
        </Label>

      </Column>
      <Column>

    </Column>
    </Columns>



      <div hidden={!show_detector_box}>

      <AiDetectorMgr
              title={"Nepi_Mgr_AI_Detector"}
          />

      </div>


      {this.renderApp()}


      <div hidden={!connected}>

        <NepiIFSaveData
          saveNamespace={appNamespace}
          title={"Nepi_IF_SaveData"}
        />

      </div>

      </Column>
      </Columns>

      )
    }  

}
export default AppAiAlerts
