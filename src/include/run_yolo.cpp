#include "run_yolo.h"

using namespace std;

run_yolo::run_yolo(const cv::String cfgfile, const cv::String weightfile, const cv::String objfile, const float confidence)
{
    this->cfg_file = cfgfile;
    this->weights_file = weightfile;
    this->obj_file = objfile;
    this->set_confidence = confidence;
    this->mydnn = cv::dnn::readNetFromDarknet(cfg_file, weights_file);
    
    this->mydnn.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    this->mydnn.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    //uncomment the below if CUDA available
    //this->mydnn.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //this->mydnn.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}
run_yolo::~run_yolo()
{

}

void run_yolo::rundarknet(cv::Mat &frame)
{
    obj_vector.clear();
    this->total_start = std::chrono::steady_clock::now();
    findboundingboxes(frame);
    this->total_end = std::chrono::steady_clock::now();
    total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    this->appro_fps = total_fps;
}

void run_yolo::display(cv::Mat frame)
{
    cv::imshow("Yolo-ed", frame);
    cv::waitKey(20);

}


void run_yolo::findboundingboxes(cv::Mat &frame)
{
    cv::Mat blob;
    double time_start = cv::getTickCount();

    blob = cv::dnn::blobFromImage(frame, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false);
    mydnn.setInput(blob);
    vector<cv::String> net_outputNames;
    net_outputNames = mydnn.getUnconnectedOutLayersNames();
    vector<cv::Mat> netOutput;
    mydnn.forward(netOutput, net_outputNames);
    findwhichboundingboxrocks(netOutput, frame);

    double time_end = cv::getTickCount();
    double elapse = (time_end-time_start)/cv::getTickFrequency();


    cout<<"time: "<<elapse*1000<<" ms"<<endl;
    cout<<"freq: "<<1/elapse<<endl;
    
}

void run_yolo::findwhichboundingboxrocks(vector<cv::Mat> &netOutput, cv::Mat &frame)
{
    vector<float> confidenceperbbox;
    vector<int> indices;
    vector<cv::Rect> bboxes;
    vector<string> classnames;
    vector<int> classids;

    getclassname(classnames);

    int indicator =0;
    for(auto &output: netOutput)
    {
        for(int i=0;i<output.rows;i++)
        {
            auto isthereanobjectconfidence = output.at<float> (i,4);
            if(isthereanobjectconfidence>set_confidence)
            {
                auto x =output.at<float>(i,0) * frame.cols;
                auto y =output.at<float>(i,1) * frame.rows;
                auto w =output.at<float>(i,2) * frame.cols;
                auto h =output.at<float>(i,3) * frame.rows;

                auto x_ = int(x - w/2);
                auto y_ = int(y - h/2);
                auto w_ = int(w);
                auto h_ = int(h);
                cv::Rect Rect_temp(x_,y_,w_,h_);

                for(int class_i=0;class_i<classnames.size();class_i++)
                {
                    auto confidence_each_class = output.at<float>(i, 5+class_i); 
                    if(confidence_each_class>set_confidence)
                    {
                        bboxes.push_back(Rect_temp);
                        confidenceperbbox.push_back(confidence_each_class);
                        classids.push_back(class_i);
                    }
                }
            }
        }
    }

    cv::dnn::NMSBoxes(bboxes,confidenceperbbox,0.1,0.1,indices);
    for(int i =0 ; i < indices.size();i++)
    {
        int index = indices[i];

        int final_x, final_y, final_w, final_h;
        final_x = bboxes[index].x;
        final_y = bboxes[index].y;
        final_w = bboxes[index].width;
        final_h = bboxes[index].height;
        cv::Scalar color;




        int temp_iy = 0;


        string detectedclass = classnames[classids[index]];
        float detectedconfidence = confidenceperbbox[index]*100;


        char temp_confidence[40];
        sprintf(temp_confidence, "%.2f", detectedconfidence);     


        string textoutputonframe = detectedclass + ": " + temp_confidence + "%";


        cv::Scalar colorforbox(rand()&255, rand()&255, rand()&255);

        cv::rectangle(frame, cv::Point(final_x, final_y), cv::Point(final_x+final_w, final_y+final_h), colorforbox,2);
        cv::putText(frame, textoutputonframe, cv::Point(final_x,final_y-10),cv::FONT_HERSHEY_COMPLEX_SMALL,1,CV_RGB(255,255,0));
        obj.confidence = detectedconfidence;
        obj.classnameofdetection = detectedclass;
        obj.boundingbox = cv::Rect(cv::Point(final_x, final_y), cv::Point(final_x+final_w, final_y+final_h));
        obj.frame = frame;
        obj_vector.push_back(obj);
    }

}

void run_yolo::getclassname(vector<std::string> &classnames)
{
    ifstream class_file(obj_file);
    if (!class_file)
    {
        cerr << "failed to open classes.txt\n";
    }

    string line;
    while (getline(class_file, line))
    {
        classnames.push_back(line);
    }
}


