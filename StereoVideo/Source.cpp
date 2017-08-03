#ifdef _DEBUG
//Debug���[�h�̏ꍇ
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300d.lib")            // opencv_core
#else
//Release���[�h�̏ꍇ
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300.lib")

#pragma comment(lib, "C:\\Program Files\\Anaconda3\\libs\\python35.lib")
#endif

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

int main(){
	Mat Q;
	Mat mapx_l, mapy_l;
	Mat mapx_r, mapy_r;
	//xml�t�@�C������K�v�ȃ}�b�v�f�[�^��ǂݍ���
	FileStorage cvfsr("C:\\Users\\0133752\\Desktop\\StereoCalibrate.xml", FileStorage::READ);
	FileNode node(cvfsr.fs, NULL);
	read(node["mapx_l"], mapx_l);
	read(node["mapy_l"], mapy_l);
	read(node["mapx_r"], mapx_r);
	read(node["mapy_r"], mapy_r);
	read(node["Q"], Q);


	VideoCapture capture_l(1), capture_r(2);

	cv::Mat image_l;
	cv::Mat image_r;

	int minDisparity = 0;
	int numDisparities = 64;
	int blockSize = 11;

	//�X�e���IBM�̃C���X�^���X��
	//StereoBM��create���\�b�h�Ń|�C���^���擾
	//���̍ۂɊe��p�����[�^�������ɓ���
	//StereoSGBM::create(
	//	minDisparity---------- - int�A�ŏ������l�A�V�t�g�i���s�ړ��j�̂Ȃ��ꍇ�ɂ�0�ŗǂ�
	//	numDisparities-------- - int�A�ő压���l�ƍŏ������l�̍��A16�̔{���ɂ���
	//	blockSize------------ - int�AOpenCV2.4�ł́ASADWindowSize�ƌĂ΂�Ă����A3�`11�̊�AStereoBM�ƈقȂ�A3����g�p�ł���
	//	P1 = 0 --------------int�A�f�t�H���g�̂܂܂ŗǂ�����
	//	P2 = 0 -------------- - int�A ����
	//	disp12MaxDiff = 0 ------int�A ���E�̎����̋��e�ő�l�A�f�t�H���g�̓`�F�b�N�Ȃ��A�f�t�H���g�ŗǂ�����
	//	preFilterCap = 0 ------ - int�A ���O�Ƀt�B���^�ő傫�Ȏ������N���b�v����A�f�t�H���g�ŗǂ�����
	//	uniquenessRatio = 0 ----int�A�ړI�֐��l�̎��_�Ƃ̍��́��䗦�A0�͔�r���Ȃ��ӁA�f�t�H���g�ŏ\���ł�����
	//	speckleWindowSize = 0 --int�A���������_��m�C�Y�������t�B���^�̃T�C�Y�A�f�t�H���g��0�͎g�p���Ȃ��ӁA����͎g���Ĕ��Ɍ��ʂ��������B
	//	speckleRange = 0 ------int�A��L�t�B���^���g�p����Ƃ��́A�����̍ő�l�A1�`2�������ŁA16�{�����A1���ǂ�����
	//	mode = StereoSGBM::MODE_SGBM----int�AOpenCV2.4�ł́AfullDP = false�Ƃ���Ă������́A�t���X�P�[���̂Q�p�X�E�_�C�i�~�b�N�v���O���~���O�����s������ɂ́AStereoSGBM::MODE_HH���g�p
	//)
	cv::Ptr<cv::StereoSGBM> ssgbm = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize);
	ssgbm->setSpeckleWindowSize(200);
	ssgbm->setSpeckleRange(1);

	while (1){
		//���E�摜�ǂݍ���
		capture_l >> image_l;
		capture_r >> image_r;

		//�X�e���I�J�����p�̕␳�}�b�v��p���ē��͉摜��␳
		remap(image_l, image_l, mapx_l, mapy_l, INTER_LINEAR);
		remap(image_r, image_r, mapx_r, mapy_r, INTER_LINEAR);

		cvtColor(image_l, image_l, CV_BGR2GRAY);
		cvtColor(image_r, image_r, CV_BGR2GRAY);

		imshow("image_l", image_l);
		imshow("image_r", image_r);

		//���E�摜����[�x�}�b�v���쐬
		cv::Mat disparity;    //((cv::MatSize)leftImg.size, CV_16S);
		ssgbm->compute(image_l, image_r, disparity);

		Mat _3dImage;
		reprojectImageTo3D(disparity, _3dImage, Q);


		//�[�x�}�b�v�����o�I�ɕ�����悤�Ƀs�N�Z���l��ϊ�
		cv::Mat disparity_map;
		double min, max;
		//�[�x�}�b�v�̍ŏ��l�ƍő�l�����߂�
		cv::minMaxLoc(disparity, &min, &max);
		//CV_8UC1�ɕϊ��A�ő僌���W��0�`255�ɂ���
		disparity.convertTo(disparity_map, CV_8UC1, 255.0 / (max - min), -255.0 * min / (max - min));
		cv::imshow("result", disparity_map);

		if (cv::waitKey(10) == 27){
			break;
		}
	}

	return 0;
}