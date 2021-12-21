#define _USE_MATH_DEFINES

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <cmath>
#include<algorithm>

using namespace cv;
using namespace std;

Mat oImgs[6], DoGs[6],resizeDogs[6],img_k1, img_k2,iX[6],iY[6],iD[6],iA[6];

vector < array<int, 2>> corrPoints;


float k1_data[3][3] = { {1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f}, {1.0f / 8.0f, 1.0f / 4.0f, 1.0f / 8.0f}, {1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f} };
float k2_data[5][5] = { {1.0f / 273.0f, 4.0f / 273.0f, 7.0f / 273.0f, 4.0f / 273.0f, 1.0f / 273.0f},
                                    {4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f},
                                    {7.0f / 273.0f, 26.0f / 273.0f, 41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f},
                                    {4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f},
                                    {1.0f / 273.0f, 4.0f / 273.0f, 7.0f / 273.0f, 4.0f / 273.0f, 1.0f / 273.0f}
};
Mat k_1 = Mat(3, 3, CV_32FC1, k1_data);
Mat k_2 = Mat(5, 5, CV_32FC1, k2_data);

vector < vector<array<int, 2>>> maxPointCloud;

vector<array<int, 2>> keyPoints;
vector<Mat> Sift_Features1, Sift_Features2;

void genOctImages(Mat& img) {

    for (int i = 0; i < 6; i++) {
        resize(img, oImgs[i], Size(img.cols / pow(2, i), img.rows / pow(2, i)));
    }
}

void genDoG(Mat& img,Mat& img_dog) {
    filter2D(img, img_k1, -1, k_1);
    filter2D(img, img_k2, -1, k_2);
    subtract(img_k1, img_k2, img_dog);

    double meanDoG = mean(img_dog)[0];
    
    for (int y = 0; y < img_dog.rows; y++) {
        for (int x = 0; x < img_dog.cols; x++) {
            if (img_dog.ptr<uchar>(y)[x] < meanDoG)
                img_dog.ptr<uchar>(y)[x] = 0;
        }
    }
}

vector<array<int,2>> genMaxPoints(Mat& img) {
   
    double maxVal;
   
    Point maxIdx;
    vector<array<int,2>> maxPoints;
        for (int y = 2; y < img.rows - 2; y = y + 5) {
            for (int x = 2; x < img.cols - 2; x = x + 5) {
                minMaxLoc(img(Range(y - 2, y + 3), Range(x - 2, x + 3)), NULL, &maxVal, NULL, &maxIdx);
                
                maxPoints.push_back({ maxIdx.x + x , maxIdx.y + y });

            }
    }
        return maxPoints;
}

void getIx(Mat& img, Mat& ixMat) {
   
    ixMat.create(img.rows, img.cols, CV_32FC1);
    
    for (int y = 1; y < img.rows - 1; y++) {
        for (int x = 1; x < img.cols - 1; x++) {
            ixMat.ptr<float>(y)[x] = (img.ptr<uchar>(y + 1)[x] - img.ptr<uchar>(y - 1)[x]) / 2;
        }
    }
}

void getIy(Mat& img, Mat& ixMat) {
    
    ixMat.create(img.rows, img.cols, CV_32FC1);
    for (int y = 1; y < img.rows - 1; y++) {
        for (int x = 1; x < img.cols - 1; x++) {
        ixMat.ptr<float>(y)[x] = (img.ptr<uchar>(y)[x+1] - img.ptr<uchar>(y)[x-1]) / 2;
        }
    }
}

void getId(Mat& ix, Mat& iy,Mat& id) {
    id.create(ix.rows, ix.cols, CV_32FC1);
    for (int y = 0; y < ix.rows; y++) {
        for (int x = 0; x < ix.cols; x++) {
            if (ix.ptr<float>(y)[x] != 0) {
                id.ptr<float>(y)[x] = atan2(iy.ptr<float>(y)[x] ,ix.ptr<float>(y)[x])* (180 / M_PI);
                
            }
            else { 
                if (iy.ptr<float>(y)[x] > 0) {
                    id.ptr<float>(y)[x] = 90;
                } else id.ptr<float>(y)[x] = -90;
            }
        }
    }
}

int approx(int num) {
    int angles[8] = { 0,45,90,135,180,-135,-90,-45 };
    int c = angles[0];
    for (int i = 0; i < 8; i++) {
        if (abs(num - angles[i]) < abs(num - c)) {
            c = angles[i];
        }
    }
    return int(distance(angles,find(angles,angles+8,c)));
}

void getIa(Mat& img,Mat& iA) {
    iA.create(img.rows, img.cols, CV_32FC1);
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            iA.ptr<float>(y)[x] = (float)approx(img.ptr<float>(y)[x]);
        }
    }
}
vector<int> normHist(int arr[8]) {
   int maxIndex = int(distance(arr, max_element(arr, arr + 8)));
   vector<int> norHist;
   
   for (int i = maxIndex; i <8; i++) {
       norHist.push_back(arr[i]);

   }
   for (int j = 0; j < maxIndex; j++) {
       norHist.push_back(arr[j]);
   }

   
   return norHist;
}

vector<Mat> getHist(Mat img[6], vector<array<int, 2>>& keyPoints) {
    Mat keyPointMat;
    vector<Mat> result;
    for (int i = 0; i <size(keyPoints); i++) {
        
        //cout << keyPoints.at(356)[0] <<"   "<< keyPoints.at(356)[1]<<"  "<<imgRows<<"   "<<img[3].rows<< endl;
 
        for (int t = 0; t < 6; t++) {

            
            int xP = int(keyPoints.at(i)[0] / pow(2,t));
            int yP = int(keyPoints.at(i)[1] / pow(2, t));
           
            if (xP >= 3 &&  yP >= 3 
                &&xP < img[t].cols - 3 && yP < img[t].rows - 3
                ) {
                //cout <<"p : "<<p<< "kp : " << i <<" xP: "<< keyPoints.at(i)[0] << " yP: " << keyPoints.at(i)[1] << " in image : " << t <<" of Rows : "<< img[t].rows<<" and Cols : "<< img[t].cols<< endl;
                //cout << i << "-" << t << "-" << p << "-" << xP << "-" << yP <<"  "<<img[t].rows << endl;
                Mat temp;
                vector<vector<int>> histArr;
                keyPointMat=Mat(6, 8, CV_32FC1);
                img[t](Rect(xP - 3, yP - 3, 7, 7)).copyTo(temp);
                int i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0;
                int iVal[8] = { 0,0,0,0,0,0,0,0 };
                int finalArr[8];
                for (int m = 0; m < temp.rows; m++) {
                    for (int n = 0; n < temp.cols; n++) {
                        float val = temp.ptr<float>(m)[n];
                        if (val == 0) {
                            i0++;
                            iVal[0]++;
                        }
                        else if (val == 1) {
                            i1++;
                            iVal[1]++;
                        }
                        else if (val == 2) {
                            i2++;
                            iVal[2]++;
                        }
                        else if (val == 3) {
                            i3++;
                            iVal[3]++;
                        }
                        else if (val == 4) {
                            i4++;
                            iVal[4]++;
                        }
                        else if (val == 5) {
                            i5++;
                            iVal[5]++;
                        }
                        else if (val == 6) {
                            i6++;
                            iVal[6]++;
                        }
                        else if (val == 7) {
                            i7++;
                            iVal[7]++;
                        }

                        vector<int>v =normHist(iVal);
                        copy(v.begin(), v.end(), finalArr);
                        
                        histArr.push_back({ i0,i1,i2,i3,i4,i5,i6,i7 });
                        
                    }
                }
                for (int c = 0; c < 8; c++) {
                    //cout << t << "   " << c << "   " << iVal[c] << endl;
                    keyPointMat.ptr<float>(t)[c] = (float)finalArr[c];
                }
                
                result.push_back(keyPointMat);
                
                //cout << size(histArr )<< endl;
              // cout << keyPointMat << endl;
                //Mat keyPointMat=Mat(6, 8, CV_32FC1, histArr);
                /*cout << "output......................" << endl;
                cout << i0 << "  " << i1 << "  " << i2 << "  " << i3 << "  " << i4 << "  " << i5 << "  " << i6 << "  " << i7 << "  " << endl;*/
            }
        }
        
    }
    return result;
}

double getNCC(Mat& img1, Mat& img2)
{
    Scalar avg1 = mean(img1);
    Scalar avg2 = mean(img2);


    Mat new_img1 = img1 - avg1[0];

    Mat new_img2 = img2 - avg2[0];

    Mat mag1 = new_img1.mul(img1);
    double sum1 = sqrt(sum(mag1)[0]);

    Mat mag2 = new_img2.mul(img2);
    double sum2 = sqrt(sum(mag2)[0]);

    double inner_product = sum(new_img1.mul(new_img2))[0];

    double ncc = (inner_product / (sum1 * sum2));
    return ncc;
}

vector<Mat> getSift(Mat& img) {
    vector<Mat> result;
    result.clear();
    int imgRows = img.rows;
    maxPointCloud.clear();
    keyPoints.clear();
    
    
    genOctImages(img);

    for (int i = 0; i < 6; i++) {
     genDoG(oImgs[i], DoGs[i]);     
    }
    for (int i = 0; i < 6; i++) {
        resize(DoGs[i], resizeDogs[i], Size(img.rows, img.cols));
    }

    for (int i = 0; i < 6; i++) {
        maxPointCloud.push_back(genMaxPoints(resizeDogs[i]));
    }
    
    for (int i = 0; i < size(maxPointCloud[5]); i++) {
        int xRef = maxPointCloud[5].at(i)[0];
        int yRef = maxPointCloud[5].at(i)[1];
        if (maxPointCloud[0].at(i)[0] == xRef && maxPointCloud[0].at(i)[1] == yRef
            && maxPointCloud[1].at(i)[0] == xRef && maxPointCloud[1].at(i)[1] == yRef
            && maxPointCloud[2].at(i)[0] == xRef && maxPointCloud[2].at(i)[1] == yRef
            && maxPointCloud[3].at(i)[0] == xRef && maxPointCloud[3].at(i)[1] == yRef
            && maxPointCloud[4].at(i)[0] == xRef && maxPointCloud[4].at(i)[1] == yRef
            && maxPointCloud[5].at(i)[0]== xRef && maxPointCloud[5].at(i)[1] == yRef
            ) {
        
            keyPoints.push_back({ xRef,yRef }); 
        }  
    }

    for (int i = 0; i < 6; i++) {
        getIx(oImgs[i], iX[i]);
    }
    for (int i = 0; i < 6; i++) {
        getIy(oImgs[i], iY[i]);
    }
    for (int i = 0; i < 6; i++) {
        getId(iX[i], iY[i], iD[i]);
    }
    for (int i = 0; i < 6; i++) {
        getIa(iD[i], iA[i]);
    }
    
    return getHist(iA, keyPoints);
}


int main(int argc, char** argv)
{
   
    cout << "Generating the SIFT Features for Given Images" << endl;
    Mat img01 = imread("C:/Sem01_Fall2021/Advanced Computer Vision/SIFTGeometricConstraintMatching/img_1_1.jpg", 0);
    Mat img02= imread("C:/Sem01_Fall2021/Advanced Computer Vision/SIFTGeometricConstraintMatching/img_1_2.jpg", 0);
    
    Sift_Features1= getSift(img01);
    int kp1 = size(keyPoints);
    int sf1 = size(Sift_Features1);
    cout << size(keyPoints) << endl;
    cout << size(Sift_Features1) << endl;

    Sift_Features2= getSift(img02);
    int kp2 = size(keyPoints);
    int sf2 = size(Sift_Features2);
    cout << size(keyPoints) << endl;
    cout << size(Sift_Features2) << endl;

    int sizeSF = 0;
    if (sf1 < sf2) {
        sizeSF = sf1;
    }
    else { sizeSF = sf2; }

        //int matchings = 0;
        for (int i = 0; i < sizeSF; i++) {
            
            double ncc = getNCC(Sift_Features1[i], Sift_Features2[i]);
            if (ncc > 0.7 && ncc < 1) {
                //matchings++;
                cout << ncc << endl;
                continue;

            }
    }
       /* int matchPercent = 0;
        if (kp1 < kp2) {
            matchPercent = (matchings / kp1) * 100;
        }
        else {
            matchPercent = (matchings / kp2) * 100;
        }
    
    cout << "Matching Percentage using NCC between Two Features is : " <<matchPercent<< endl;*/
   
 
    namedWindow("img01", WINDOW_NORMAL);
    namedWindow("img02", WINDOW_NORMAL);

    while (1)
    {
        imshow("img01", img01);
        imshow("img02", img02);

        
        char c = waitKey(1);
        if (c == 27)
            break;
    }
    return 1;
}