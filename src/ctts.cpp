#include "include/run_yolo.h"
#include <fstream>
#include <iostream>
#include <string>
#include "include/json/json/json.h"
#include "include/json/jsoncpp.cpp"
#include "curl/curl.h"
#include "rapidjson/schema.h"
#include "rapidjson/document.h"
static int test;

static int counter = 1;
static cv::String weightpath ="/home/wonder/ctts/src/include/yolo/yolov4-tiny.weights";
static cv::String cfgpath ="/home/wonder/ctts/src/include/yolo/yolov4-tiny.cfg";
static cv::String classnamepath = "/home/wonder/ctts/src/include/yolo/coco.names";
static run_yolo Yolonet(cfgpath, weightpath, classnamepath, float(0.5));

static string plate_result = "hello";
static string image_path;

static cv::VideoWriter output
("/home/wonder/ctts/images/video.avi", 
cv::VideoWriter::fourcc('M','J','P','G'), 30, cv::Size(1920,1080));

using namespace std;
using namespace rapidjson;


namespace
{
	size_t callback	(const char* in, std::size_t size, std::size_t num, std::string* out)
	{
		const size_t totalBytes(size * num);
		out->clear();
		out->append(in, totalBytes);
		return totalBytes;
	}
}

void getplateno(string fileName, string mode)
{
	curl_global_init(CURL_GLOBAL_ALL);

	curl_off_t speed_upload, total_time;
	curl_mime *form = NULL;
	curl_mimepart *field = NULL;

	CURL *hnd = curl_easy_init();

	//curl_easy_setopt(hnd, CURLOPT_CUSTOMREQUEST, "POST");
	curl_easy_setopt(hnd, CURLOPT_URL, "http://localhost:8080/v1/plate-reader/");

	form = curl_mime_init(hnd); //initialize form fields

	/* Fill in the file upload field */
	field = curl_mime_addpart(form);
	curl_mime_name(field, "upload");
	curl_mime_filedata(field, fileName.c_str());

	// Add more fields
	// config mode
	if(mode.length()) {
        curl_mimepart *part = NULL;
        part = curl_mime_addpart(form);
        curl_mime_name(part, "config");

        if (strcmp(mode.c_str(),"redacted") == 0){
            curl_mime_data(part, "{\"mode\":\"redacted\"}", CURL_ZERO_TERMINATED);
        } else if (strcmp(mode.c_str(),"fast") == 0){
            curl_mime_data(part, "{\"mode\":\"fast\"}", CURL_ZERO_TERMINATED);
        } else{
            cout << "Unknown config mode : "+mode+"\n Valid Options: fast, redacted\n";
            exit(1);
        }
    }

	curl_easy_setopt(hnd, CURLOPT_MIMEPOST, form);

	struct curl_slist *headers = NULL;
	// headers = curl_slist_append(headers, "cache-control: no-cache");
	// headers = curl_slist_append(headers, ("Authorization: Token "+ auth_token).c_str());
	// curl_easy_setopt(hnd, CURLOPT_HTTPHEADER, headers);

	unique_ptr<std::string> httpData(new std::string()); //initializing string pointer to  get data
	// Hook up data handling function.
	curl_easy_setopt(hnd, CURLOPT_WRITEFUNCTION, callback);

	// Hook up data container (will be passed as the last parameter to the
	// callback handling function).  Can be any pointer type, since it will
	// internally be passed as a void pointer.
	curl_easy_setopt(hnd, CURLOPT_WRITEDATA, httpData.get());

	CURLcode ret = curl_easy_perform(hnd); //perform the request

	if (ret != CURLE_OK) { //failed case
		fprintf(stderr, "curl_easy_perform() failed: %s\n",
			curl_easy_strerror(ret));
	}
	else {
		/* now extract transfer info */
		curl_easy_getinfo(hnd, CURLINFO_SPEED_UPLOAD_T, &speed_upload);
		curl_easy_getinfo(hnd, CURLINFO_TOTAL_TIME_T, &total_time);

		fprintf(stderr, "Speed: %" CURL_FORMAT_CURL_OFF_T " bytes/sec during %"
			CURL_FORMAT_CURL_OFF_T ".%06ld seconds\n",
			speed_upload,
			(total_time / 1000000), (long)(total_time % 1000000));
	}

	curl_easy_cleanup(hnd);
	curl_mime_free(form);

	Json::Value jsonData;
	Json::Reader jsonReader;

	if (jsonReader.parse(*httpData, jsonData))
	{
		cout << "succeed" << endl;

		Json::FastWriter writer;//write json_result 2 string
		string test;
		std::string json_result;
		json_result = writer.write(jsonData);
	
		char *temp = new char[json_result.length() + 1];
		strcpy(temp, json_result.c_str());//write string 2 char*
		
		rapidjson::Document doc;
		doc.Parse(temp);//parse char* to rapidjson::document

		for(auto& result : doc["results"].GetArray())
		//use rapidjson's function to extract plate number.
		{
			for(auto& what : result.GetObject())
			{
				string final_result(what.name.GetString(), what.name.GetStringLength());
				if(final_result == "plate")
				{
					string lp_temp = what.value.GetString();
					plate_result.clear();
					for(size_t i = 0; i < lp_temp.size(); i++)
					{
						if( i > 1)
						{
							char show = lp_temp[i];
							if(show > 96 && show < 123)
							{
								if(show == 105)
								{
									plate_result.push_back('1');
								}
								else if(show == 115)
								{
									plate_result.push_back('5');								
								}
								else if(show == 111)
								{
									plate_result.push_back('0');								
								}
								else
								{
									plate_result.push_back(' ');
								}						
							}
							else
								plate_result.push_back(lp_temp[i]);
						}
						else
						{
							char show = lp_temp[i];
							if(show > 47 && show < 58)
							{
								if(show == 49)
								{
									plate_result.push_back('i');
								}
								else if(show == 53)
								{
									plate_result.push_back('s');								
								}
								else if(show == 48)
								{
									plate_result.push_back('o');								
								}
								else
								{
									plate_result.push_back(' ');
								}						
							}
							else
								plate_result.push_back(lp_temp[i]);
						}
					}
				}
			}
		}

		delete [] temp;
	}
	else
	{
		std::cout << "Could not parse HTTP data as JSON" << std::endl;
		std::cout << "HTTP data was:\n" << *httpData.get() << std::endl;
	}

}

void execute(cv::Mat& frame, timer clock)
{
	Yolonet.rundarknet(frame);

	vector<objectinfo> savethings;
	for(auto what : Yolonet.obj_vector)
	{
		if(what.classnameofdetection == "truck")
		{
			savethings.push_back(what);
		}
	}
	if(savethings.size()>=1)
	{
		cv::imwrite("/home/wonder/ctts/images/read.png", frame);
		cv::imwrite("/home/wonder/ctts/images/temp.png", frame);
		if(counter%10 == 0)
		{
			getplateno("/home/wonder/ctts/images/read.png", "fast");	
			cout<<"plate no.: " <<plate_result<<endl;
		}

		// if(clock.timesup())
		// {
		// 	clock.reset();
		// 	cout<<endl<<"hi"<<" missisipi"<<endl<<endl;;
			
		// 	getplateno("/home/wonder/ctts/images/read.png", "fast");	
		// 	cout<<"plate no.: " <<plate_result<<endl;
		// }
		// else
		// {
		// 	// capture >> car;
		// 	// cv::imshow("hi", car);
		// 	// cv::waitKey(20);
		// }
	}
	counter++;
	cv::putText(frame, plate_result, cv::Point(1400,750),
	cv::FONT_HERSHEY_PLAIN, 7, CV_RGB(255,0,0),4);
	Yolonet.display(frame);
	output.write(frame);
}

void greetings()
{
	cout<<endl<<endl;
	cout<<"09 Dec 2021 Demo @SaiSha, New Tettitory, Hong Kong"<<endl<<endl;
	std::vector<std::string> v;

	v.push_back("WWWWW    W   W   W   W              WWWWW  WWWWW   WWWWW   WWWWW");
	v.push_back("W        W   W   W  W               W        W       W     W    ");
	v.push_back("W        W   W   W W                W        W       W     W    ");
	v.push_back("WWWWW    WWWWW   WW      WWWWWWW    W        W       W     WWWWW");
	v.push_back("    W    W   W   W W                W        W       W         W");
	v.push_back("    W    W   W   W  W               W        W       W         W");
	v.push_back("WWWWW    W   W   W   W              WWWWW    W       W     WWWWW");

	for (auto row : v)
	{
		for (auto col : row)
			cout << col;
		cout << endl;
	}
	cout<<endl<<endl;
	cout<<"copyright @Wonder Construct, HKCRC"<<endl<<endl;
	cout<<"------------------------------------------------------------------------"<<endl<<endl;;;
}

int main()
{
	greetings();
	cv::Mat car;
	cv::VideoCapture capture("final.mp4");
	if (!capture.isOpened()) 
	{
        cout << "Error opening video stream or file" << endl;
        return -1;
    }


    double dt = 0.25;
	timer clock(dt);
	clock.reset();

	while(true)
	{
		capture >> car;
		cout<<car.size<<endl;
		execute(car, clock);

	}
	
	getplateno("/home/wonder/ctts/images/read.png", "fast");
	cout<<plate_result<<endl;	






	// ofstream file;
	// file.open("response.txt", ios::app);
	// file << NULL;
	// file << "\n\n";
	// file.close();
	// cout << "\n\n";
	return 0;
}
