#pragma once

#define NORM_WIDTH    256                                // 规格化宽
#define NORM_HEIGHT   256                                // 规格化高
#define SUB_NORM      8                                  // 分块DCT变换窗口边长
#define SUB_WIDTH     SUB_NORM                           // 分块DCT变换窗口宽
#define SUB_HEIGHT    SUB_NORM                           // 分块DCT变换窗口高
#define SUB_AREA      SUB_HEIGHT * SUB_WIDTH             // 窗口面积
#define ABS(a)       ((a) < 0 ? -(a) : (a))
#define IMG_DBS       enum DB{TID2013, TID2008, LIVE2, IVC, CSIQ, USCSIPI, UCID, koniq_10k, CLIVE}       // 使用的图像数据库种类

class R_ARRAY
{
public:
	R_ARRAY() {
		S0 = (double*)malloc(sizeof(double) * (NORM_WIDTH * NORM_HEIGHT / SUB_AREA));
		S1 = (double*)malloc(sizeof(double) * (NORM_WIDTH * NORM_HEIGHT / SUB_AREA));
		S2 = (double*)malloc(sizeof(double) * (NORM_WIDTH * NORM_HEIGHT / SUB_AREA));
		S3 = (double*)malloc(sizeof(double) * (NORM_WIDTH * NORM_HEIGHT / SUB_AREA));
		S4 = (double*)malloc(sizeof(double) * 4 * (NORM_WIDTH * NORM_HEIGHT / SUB_AREA));
		S5 = (double*)malloc(sizeof(double) * 4 * (NORM_WIDTH * NORM_HEIGHT / SUB_AREA));
		S6 = (double*)malloc(sizeof(double) * 4 * (NORM_WIDTH * NORM_HEIGHT / SUB_AREA));
		S7 = (double*)malloc(sizeof(double) * 16 * (NORM_WIDTH * NORM_HEIGHT / SUB_AREA));
		S8 = (double*)malloc(sizeof(double) * 16 * (NORM_WIDTH * NORM_HEIGHT / SUB_AREA));
		S9 = (double*)malloc(sizeof(double) * 16 * (NORM_WIDTH * NORM_HEIGHT / SUB_AREA));
	}

public:
	double* S0;
	double* S1;
	double* S2;
	double* S3;
	double* S4;
	double* S5;
	double* S6;
	double* S7;
	double* S8;
	double* S9;
};



class CIQAplatform : public CDialogEx
{
// 构造
public:
	CIQAplatform(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_IQA_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

public:
	virtual void OnDraw(CDC* pDC);  // 重写以绘制该视图

public:
	void ShowInfo(CDC* pDC);
	afx_msg void OnOpencvLoadReferenceImage();
	afx_msg void OnOpencvLoadDistortedImage();
	afx_msg void OnRRGSM();
	afx_msg void Img_Gradient_Map();

public:
	void BGRBlkDCT(Mat img, R_ARRAY s);
	std::vector<Mat> dct10subbands(R_ARRAY RDCTArray, Mat RDCT, Mat RDCT1);
	Mat ImgSRDFT(Mat image);
	Mat ImgSRDLC(Mat image);
	Mat ImgSRDHC(const Mat& img3f);
	double calc3orderMom(Mat& channel);
	void colorMom(Mat& img, double* Mom);
	CString ColorMomentRef(Mat RefImg, double* Mom1);
	double* TenengradMeasure(double* score, Mat image);
	double NRSSMeasure(Mat img);

public:
	void meshgrid(Mat& meshX, Mat& meshY, double xb, double xe, double yb, double ye);
	Mat fft2(Mat input_img);
	Mat ifft2(Mat input_img);
	Mat circshift(Mat in, const Point& delta);
	Mat fftshift(Mat in);
	Mat ifftshift(Mat in);
	Mat ireal(Mat in);
	Mat Make_CSF(Mat in);
	Mat Img_Gradient_Map(Mat in);
	/*MatrixXcd eigen3fft2(Mat input_img) {
		const int x = input_img.rows;
		const int y = input_img.cols;
		FFT<double> fft;
		std::unique_ptr<MatrixXd> in = std::make_unique<MatrixXd>(x, y);
		MatrixXcd fftout(x, y);
		cv2eigen(input_img, *in);
		for (int i = 0; i < x; i++) {
			VectorXcd tmpv(y);
			fft.fwd(tmpv, in->row(i));
			fftout.row(i) = tmpv;
		}
		for (int i = 0; i < y; i++) {
			VectorXcd tmpv(x);
			fft.fwd(tmpv, fftout.col(i));
			fftout.col(i) = tmpv;
		}
		return fftout;
	}*/
	afx_msg void OnBnClickedButtonDg();
	afx_msg void OnBnClickedButtonHc();
	afx_msg void OnBnClickedButtonIg2();
	afx_msg void OnBnClickedButtonHc2();
	afx_msg void OnBnClickedOk();
};

class CProcDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CProcDlg)

public:
	CProcDlg(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CProcDlg();
	void OnCancel();
	void PostNcDestroy();

	// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG1 };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:

	CProgressCtrl MPR; /* 进度条 */
};


typedef struct _IQAPCDlg {
	CIQAplatform* cia;
	CProcDlg* cpd;
}*pIQA;
