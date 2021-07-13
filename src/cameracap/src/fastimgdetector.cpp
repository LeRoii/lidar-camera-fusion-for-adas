#include "fastimgdetector.h"

fasterimgdetector::fasterimgdetector(ros::NodeHandle& nh):it(nh)
{
    img_pub = it.advertise("/YOLO_detect_result", 1);
    objmsg_pub = nh.advertise<objmsg::objArray>("/img_obj",100);
}

fasterimgdetector::~fasterimgdetector()
{
    delete m_pdetector;
    m_pdetector = nullptr;
}

void fasterimgdetector::initDetector(std::string net, int n_classes, float conf_thresh)
{
    m_pdetector = new tk::dnn::Yolo3Detection();
    bool ret = m_pdetector->init(net, n_classes, 1, conf_thresh);
    ROS_DEBUG_STREAM("detNN init okkkkk"<<ret);
}

void fasterimgdetector::imgcallback(const sensor_msgs::Image &msg)
{
    ROS_INFO_STREAM("fast det Callback");
    printf("msg time:%f\n", msg.header.stamp.toSec());
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat img = cv_ptr->image.clone();

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;
    std::vector<int> detret;
    std::vector<std::string> classnames;

    // cv::Mat imgg = cv::imread("/space/data/1.jpg");
    batch_frame.push_back(img);
    batch_dnn_input.push_back(img.clone());
    m_pdetector->update(batch_dnn_input, 1);

    ROS_INFO_STREAM("fast det Callback update fini");
    m_pdetector->draw(batch_frame, detret, classnames);
    // m_pdetector->draw(batch_frame);

    std::vector<objmsg::obj> boxes;
    objmsg::objArray boxesMsg;
    for(int i=0;i<detret.size();i=i+5)
    {
        objmsg::obj objbox;
        objbox.x = detret[i];
        objbox.y = detret[i+1];
        objbox.w = detret[i+2];
        objbox.h = detret[i+3];
        objbox.type = detret[i+4];
        boxes.push_back(objbox);
    }

    boxesMsg.objects = boxes;
    boxesMsg.header.stamp = msg.header.stamp;
    objmsg_pub.publish(boxesMsg);

    sensor_msgs::ImagePtr pubmsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", batch_frame.back()).toImageMsg();
    img_pub.publish(pubmsg);
}