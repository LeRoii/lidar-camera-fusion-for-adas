#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <log4cxx/logger.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>
#include <fstream>

#include "Yolo3Detection.h"
// #include "imgdetector.h"    // ori yolo
#include "fastimgdetector.h"    //fast yolo


int main (int argc, char** argv){
    ros::init(argc, argv, "camcap_node");
    ros::NodeHandle n;

    log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME)->setLevel(
    ros::console::g_level_lookup[ros::console::levels::Debug]);
    ros::console::notifyLoggerLevelsChanged();

    //*************fast detector init**********
    std::string net = "/space/model/yolo-kitti/yolo4_fp32.rt";
    int n_classes = 80;
    float conf_thresh=0.3;
    fasterimgdetector fastdetector(n);
    fastdetector.initDetector(net, n_classes, conf_thresh);

    //************fast detector end***********

    //************ori yolo init*************
    // imgdetector detector(n);
    // // std::string cfg_filename = "/space/code/jetson_bot/src/cameracap/cfg/yolov3.cfg";
    // // std::string weight_filename = "/space/model/yolov3.weights";
    // // std::string class_filename = "/space/code/jetson_bot/src/cameracap/cfg/coco.names";
    // std::string cfg_filename = "/space/model/yolo-kitti/yolo4-sq-416.cfg";
    // std::string weight_filename = "/space/model/yolo-kitti/yolo4-sq-416.weight";
    // std::string class_filename = "/space/model/yolo-kitti/sq.names";
    
    // detector.initDetector(cfg_filename, weight_filename, class_filename);
    // //************ori yolo end**************


    // ros::Subscriber Img_sub = n.subscribe("/cam_front/csi_cam/image_raw", 1, &imgdetector::imgcallback, &detector);
    // ros::Subscriber Img_fsub = n.subscribe("/cam_front/csi_cam/image_raw", 1, &fasterimgdetector::imgcallback, &fastdetector);
    ros::Subscriber Img_fsub = n.subscribe("/wideangle/image_raw", 1, &fasterimgdetector::imgcallback, &fastdetector);

    ros::spin();

    ros::Rate loop_rate(30);

    return 0;
}

