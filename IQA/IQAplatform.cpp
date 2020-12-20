
#include "pch.h"
#include "framework.h"
#include "IQA.h"
#include "IQAplatform.h"
#include "afxdialogex.h"
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <atlconv.h>
#include <iostream>
#include <io.h>
#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// platform


CIQAplatform::CIQAplatform(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_IQA_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CIQAplatform::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CIQAplatform, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON2, &CIQAplatform::OnOpencvLoadReferenceImage)
	ON_BN_CLICKED(IDC_BUTTON3, &CIQAplatform::OnOpencvLoadDistortedImage)
	ON_BN_CLICKED(IDC_BUTTON4, &CIQAplatform::OnRRGSM)
	ON_BN_CLICKED(IDC_BUTTON_IG, &CIQAplatform::Img_Gradient_Map)
	ON_BN_CLICKED(ID_BUTTON_HC, &CIQAplatform::OnBnClickedButtonHc)
	ON_BN_CLICKED(IDC_BUTTON_IG2, &CIQAplatform::OnBnClickedButtonIg2)
	ON_BN_CLICKED(ID_BUTTON_HC2, &CIQAplatform::OnBnClickedButtonHc2)
END_MESSAGE_MAP()

IMPLEMENT_DYNAMIC(CProcDlg, CDialogEx)

CProcDlg::CProcDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG1, pParent)
{
}

CProcDlg::~CProcDlg()
{
}

/* 更新进度条 */
void CProcDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_PROGRESS1, MPR);
}

/* 进度完成后关闭对话框 */
void CProcDlg::OnCancel()
{
	this->ShowWindow(SW_HIDE);
	DestroyWindow();
}
void CProcDlg::PostNcDestroy()
{
	CDialog::PostNcDestroy();
	delete this;
}
BEGIN_MESSAGE_MAP(CProcDlg, CDialogEx)
END_MESSAGE_MAP()

// 消息处理程序

BOOL CIQAplatform::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	namedWindow("RefImg");
	namedWindow("DisImg");
	namedWindow("RefSal");
	namedWindow("DSmap");
	HWND hWnd_R = (HWND)cvGetWindowHandle("RefImg");
	HWND hWnd_D = (HWND)cvGetWindowHandle("DisImg");
	HWND hWnd_SalRef = (HWND)cvGetWindowHandle("RefSal");
	HWND hWnd_DisSal = (HWND)cvGetWindowHandle("DSmap");
	HWND hParent_R = ::GetParent(hWnd_R);
	::SetParent(hWnd_R, GetDlgItem(IDC_STATIC_R)->m_hWnd);
	::ShowWindow(hParent_R, SW_HIDE);
	HWND hParent_D = ::GetParent(hWnd_D);
	::SetParent(hWnd_D, GetDlgItem(IDC_STATIC_D)->m_hWnd);
	::ShowWindow(hParent_D, SW_HIDE);
	HWND hParent_SalRef = ::GetParent(hWnd_SalRef);
	::SetParent(hWnd_SalRef, GetDlgItem(IDC_STATIC_ICFS)->m_hWnd);
	::ShowWindow(hParent_SalRef, SW_HIDE);
	HWND hParent_DisSal = ::GetParent(hWnd_DisSal);
	::SetParent(hWnd_DisSal, GetDlgItem(IDC_STATIC_SMD)->m_hWnd);
	::ShowWindow(hParent_DisSal, SW_HIDE);
	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}
void CIQAplatform::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。
//void CIQAplatform::ShowInfo(CDC* pDC) {
//	pEdit = new CEdit;
//	pEdit->Create(WS_VISIBLE | ES_LEFT | ES_MULTILINE | ES_AUTOVSCROLL | ES_AUTOHSCROLL, CRect(10, 30, 450, 450), this, WM_USER + 100);
//	pEdit->ShowScrollBar(SB_VERT, TRUE);
//	pEdit->ShowScrollBar(SB_HORZ, TRUE);
//	pEdit->ShowWindow(SW_SHOW);
//	pEdit->LineScroll(pEdit->GetLineCount());
//}
void CIQAplatform::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}
void CIQAplatform::OnDraw(CDC* pDC) {
	/*ShowInfo(pDC);*/
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CIQAplatform::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

//////
UINT Compute(LPVOID lpParam);
Mat RefImg, DisImg, ModRefImg, ModDisImg, org, dis;
Mat RICF, RIGM, DICF, DIGM;
CEdit* pEdit;

void CIQAplatform::OnOpencvLoadReferenceImage()
{
	// TODO: 在此添加控件通知处理程序代码
	CString m_strFilePath = _T("");
	LPCTSTR szFilter = _T("BMP(*.bmp)|*.bmp|JPG(*.jpg)|*.jpg|ALLSUPORTFILE(*.*)|*.*||");
	CFileDialog dlgFileOpenImg(TRUE, NULL, NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter, NULL);
	if (dlgFileOpenImg.DoModal() == IDOK)
		m_strFilePath = dlgFileOpenImg.GetPathName();
	else
		return;
	USES_CONVERSION;
	RefImg = imread(W2A(m_strFilePath.AllocSysString()));
	imshow("RefImg", RefImg);
}

void CIQAplatform::OnOpencvLoadDistortedImage()
{
	// TODO: 在此添加控件通知处理程序代码
	CString m_strFilePath = _T("");
	LPCTSTR szFilter = _T("BMP(*.bmp)|*.bmp|JPG(*.jpg)|*.jpg|ALLSUPORTFILE(*.*)|*.*||");
	CFileDialog dlgFileOpenImg(TRUE, NULL, NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter, NULL);
	if (dlgFileOpenImg.DoModal() == IDOK)
		m_strFilePath = dlgFileOpenImg.GetPathName();
	else
		return;
	USES_CONVERSION;
	DisImg = imread(W2A(m_strFilePath.AllocSysString()));
	imshow("DisImg", DisImg);
}

// 显著性区域检测

/* FT算法 */
Mat CIQAplatform::ImgSRDFT(Mat image) {

	Mat Lab;
	Mat SalImg = Mat::zeros(image.size(), CV_8UC1);

	cvtColor(image, Lab, CV_BGR2Lab);

	int MeanL = 0, Meana = 0, Meanb = 0;
	for (int i = 0; i < image.rows; i++) {
		Point3_<uchar>* data = Lab.ptr<Point3_<uchar> >(i);
		for (int j = 0; j < image.cols; j++) {
			MeanL += data[j].x;
			Meana += data[j].y;
			Meanb += data[j].z;
		}
	}
	MeanL /= (image.rows * image.cols);
	Meana /= (image.rows * image.cols);
	Meanb /= (image.rows * image.cols);

	GaussianBlur(Lab, Lab, Size(3, 3), 0, 0);

	int val;
	int max_v = 0, min_v = 1 << 28;
	for (int i = 0; i < image.rows; i++) {
		Point3_<uchar>* data = Lab.ptr<Point3_<uchar> >(i);
		uchar* sal = SalImg.ptr<uchar>(i);
		for (int j = 0; j < image.cols; j++) {
			sal[j] = sqrt((MeanL - data[j].x) * (MeanL - data[j].x) + (data[j].y - Meana) * (data[j].y - Meana) + (data[j].z - Meanb) * (data[j].z - Meanb));
			max_v = MAX(max_v, sal[j]);
			min_v = MIN(min_v, sal[j]);
		}
	}

	for (int Y = 0; Y < image.rows; Y++) {
		uchar* sal = SalImg.ptr<uchar>(Y);
		for (int X = 0; X < image.cols; X++) {
			sal[X] = (sal[X] - min_v) * 255 / (max_v - min_v);
		}
	}

	ModRefImg = SalImg;
	return SalImg;
}

/* LC算法 */
Mat CIQAplatform::ImgSRDLC(Mat image) {

	int width = image.cols, height = image.rows, curIndex = 0, value = 0;
	uchar* gray = (uchar*)malloc(width * height);
	int* histGram = (int*)malloc(256 * sizeof(int));
	int* dist = (int*)malloc(256 * sizeof(int));
	float* distMap = (float*)malloc(height * width * sizeof(float));
	memset(histGram, 0, 256 * sizeof(int));

	/* 得到每个像素灰度值 */
	for (int i = 0; i < height; i++) {
		curIndex = i * width;
		for (int j = 0; j < width; j++) {
			int b = image.at<Vec3b>(i, j)[0];
			int g = image.at<Vec3b>(i, j)[1];
			int r = image.at<Vec3b>(i, j)[2];
			value = (r * 38 + g * 75 + b * 15) >> 7;
			gray[curIndex] = value;
			histGram[value]++;
			curIndex++;
		}
	}

	/* 距离乘该灰度值频数 */
	for (int i = 0; i < 256; i++) {
		value = 0;
		for (int j = 0; j < 256; j++)  // 遍历所有灰度值,计算与Y的距离
			value += ABS(i - j) * histGram[j];
		dist[i] = value;       // 存放灰度值Y全图像素的距离
	}

	/* 将dist值归一化到0-255 */
	int mindist = dist[0], maxdist = dist[0], ddist;
	for (int k = 0; k < 256; k++) {
		if (dist[k] > maxdist)
			maxdist = dist[k];
		if (dist[k] < mindist)
			mindist = dist[k];
	}
	ddist = maxdist - mindist;
	for (int i = 0; i < 256; i++)
		dist[i] = (int)(1.0f * (dist[i] - mindist) / ddist * 256.0f);

	/* 计算全图每个像素的显著性 */
	for (int i = 0; i < height; i++) {
		curIndex = i * width;
		for (int j = 0; j < width; j++) {
			distMap[curIndex] = dist[gray[curIndex]];
			curIndex++;
		}
	}

	/* 对显著性图像SalImg像素点进行遍历赋值 */
	Mat SalImg = Mat::zeros(image.size(), CV_8UC3);
	for (int i = 0; i < height; i++) {
		curIndex = i * width;
		uchar* data = SalImg.ptr<uchar>(i);
		for (int j = 0; j < width * 3; j++) {
			data[j++] = distMap[curIndex];
			data[j++] = distMap[curIndex];
			data[j] = distMap[curIndex];
			curIndex++;
		}
	}

	ModRefImg = SalImg;
	return SalImg;
}

/* HC算法 */
Mat CIQAplatform::ImgSRDHC(const Mat& img3f) {

	static const int ClrNums[3] = { 12, 12, 12 };
	static const float ClrTmp[3] = { ClrNums[0] - 0.0001f, ClrNums[1] - 0.0001f, ClrNums[2] - 0.0001f };
	static const int w[3] = { ClrNums[1] * ClrNums[2], ClrNums[2], 1 };
	Mat idx1i, binColor3f, colorNums1i, weight1f, _colorSal;

	CV_Assert(img3f.data != NULL);
	idx1i = Mat::zeros(img3f.size(), CV_32S);
	int rows = img3f.rows, cols = img3f.cols;

	// 建立调色板
	std::map<int, int> pallet;
	for (int y = 0; y < rows; y++) {
		const Point3_<float>* imgData = img3f.ptr<Point3_<float> >(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++) {
			idx[x] = (int)(imgData[x].x * ClrTmp[0]) * w[0] + (int)(imgData[x].y * ClrTmp[1]) * w[1] + (int)(imgData[x].z * ClrTmp[2]);
			pallet[idx[x]]++;
		}
	}

	// 寻找重要颜色
	int maxNum = 0;
	{
		double ratio = 0.94999999999999999556;
		int count = 0;
		std::vector<std::pair<int, int> > num; // (num, color)对
		num.reserve(pallet.size());
		for (auto it = pallet.begin(); it != pallet.end(); it++)
			num.push_back(std::pair<int, int>(it->second, it->first)); // (color, num)对
		sort(num.begin(), num.end(), std::greater<std::pair<int, int> >());
		maxNum = (int)num.size();
		int maxDropNum = cvRound(rows * cols * (1 - ratio));
		for (int crnt = num[maxNum - 1].first; crnt < maxDropNum && maxNum > 1; maxNum--)
			crnt += num[maxNum - 2].first;
		maxNum = min(maxNum, 256); // To avoid very rarely case
		if (maxNum < 10)
			maxNum = min((int)pallet.size(), 100);
		pallet.clear();
		for (int i = 0; i < maxNum; i++)
			pallet[num[i].second] = i;

		std::vector<Vec3i> color3i(num.size());
		for (unsigned int i = 0; i < num.size(); i++) {
			color3i[i][0] = num[i].second / w[0];
			color3i[i][1] = num[i].second % w[0] / w[1];
			color3i[i][2] = num[i].second % w[1];
		}

		for (unsigned int i = maxNum; i < num.size(); i++) {
			int simIdx = 0, simVal = INT_MAX;
			for (int j = 0; j < maxNum; j++) {
				int d_ij = (color3i[i][0] - color3i[j][0]) * (color3i[i][0] - color3i[j][0]) + (color3i[i][1] - color3i[j][1]) * (color3i[i][1] - color3i[j][1]) + (color3i[i][2] - color3i[j][2]) * (color3i[i][2] - color3i[j][2]);
				if (d_ij < simVal)
					simVal = d_ij, simIdx = j;
			}
			pallet[num[i].second] = pallet[num[simIdx].second];
		}
	}

	binColor3f = Mat::zeros(1, maxNum, CV_32FC3);
	colorNums1i = Mat::zeros(binColor3f.size(), CV_32S);

	Vec3f * color = (Vec3f*)(binColor3f.data);
	int* colorNum = (int*)(colorNums1i.data);
	for (int y = 0; y < rows; y++)
	{
		const Vec3f* imgData = img3f.ptr<Vec3f>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++)
		{
			idx[x] = pallet[idx[x]];
			color[idx[x]] += imgData[x];
			colorNum[idx[x]] ++;
		}
	}
	for (int i = 0; i < binColor3f.cols; i++)
		color[i] /= colorNum[i];

	cvtColor(binColor3f, binColor3f, CV_BGR2Lab);

	normalize(colorNums1i, weight1f, 1, 0, NORM_L1, CV_32F); //归一化

	// 得到颜色的显著值
	int BinN = binColor3f.cols;
	_colorSal = Mat::zeros(1, BinN, CV_32F);
	float* colorSa = (float*)(_colorSal.data);
	std::vector<std::vector<std::pair<float, int> > > simila(BinN); // Similar color: how similar and their index
	Vec3f * color1 = (Vec3f*)(binColor3f.data);
	float* w1 = (float*)(weight1f.data);
	for (int i = 0; i < BinN; i++) {                  //获得颜色i与其他颜色的距离差
		std::vector<std::pair<float, int> >& similari = simila[i];
		similari.push_back(std::make_pair(0.f, i));
		for (int j = 0; j < BinN; j++) {
			if (i == j)
				continue;
			float dij = sqrt((color1[i][0] - color1[j][0]) * (color1[i][0] - color1[j][0]) + (color1[i][1] - color1[j][1]) * (color1[i][1] - color1[j][1]) + (color1[i][2] - color1[j][2]) * (color1[i][2] - color1[j][2]));
			similari.push_back(std::make_pair(dij, j));
			colorSa[i] += w1[j] * dij;
		}
		sort(similari.begin(), similari.end());
	}

	CV_Assert(binColor3f.size() == _colorSal.size() && _colorSal.rows == 1);

	Mat tmpSal;
	_colorSal.copyTo(tmpSal);
	float* sal = (float*)(tmpSal.data);
	float* nSal = (float*)(_colorSal.data);

	// Distance based smooth
	int n = MAX(cvRound(BinN / 4.0f), 2);
	std::vector<float> dist(n, 0), val(n);
	for (int i = 0; i < BinN; i++) {
		const std::vector<std::pair<float, int> >& similari = simila[i];
		float totalDist = 0;

		val[0] = sal[i];
		for (int j = 1; j < n; j++) {
			int ithIdx = similari[j].second;
			dist[j] = similari[j].first;
			val[j] = sal[ithIdx];
			totalDist += dist[j];
		}
		float valCrnt = 0;
		for (int j = 0; j < n; j++)
			valCrnt += val[j] * (totalDist - dist[j]);

		nSal[i] = valCrnt / ((n - 1) * totalDist);
	}

	float* colorSal = (float*)(_colorSal.data);
	Mat salHC1f(img3f.size(), CV_32F);

	for (int r = 0; r < img3f.rows; r++) {             // 图像中每个像素的显著值等于该像素颜色值的显著值
		float* salV = salHC1f.ptr<float>(r);
		int* _idx = idx1i.ptr<int>(r);
		for (int c = 0; c < img3f.cols; c++)
			salV[c] = colorSal[_idx[c]];
	}

	GaussianBlur(salHC1f, salHC1f, Size(3, 3), 0);     // 高斯模糊去噪
	normalize(salHC1f, salHC1f, 0, 1, NORM_MINMAX);    // 归一化

	return salHC1f;
}


// 以步长为1创建网格
void CIQAplatform::meshgrid(Mat& meshX, Mat& meshY, double xb, double xe, double yb, double ye) {
	int rows = xe - xb, cols = ye - yb;
	Mat_<double> vecX(1, rows + 1);
	Mat_<double> vecY(cols + 1, 1);
	for (int i = 0; i <= rows; i++) {
		vecX(i) = xb;
		xb = xb + 1.0;
	}
	for (int i = 0; i <= cols; i++) {
		vecY(i) = yb;
		yb = yb + 1.0;
	}
	for (int i = 0; i <= cols; i++) {
		vecX.copyTo(meshX.row(i));
	}
	for (int i = 0; i <= rows; i++) {
		vecY.copyTo(meshY.col(i));
	}
}

Mat CIQAplatform::fft2(Mat input_img) {
	Mat Fourier;
	Mat planes[] = { Mat_<double>(input_img), Mat::zeros(input_img.size(), CV_64F) };
	merge(planes, 2, Fourier);
	dft(Fourier, Fourier);
	return Fourier;
}

Mat CIQAplatform::ifft2(Mat in) {
	Mat ICFcx;
	idft(in, ICFcx, DFT_COMPLEX_OUTPUT + DFT_SCALE);
	return ICFcx;
}

Mat CIQAplatform::ireal(Mat in) {
	std::vector<Mat> iv;
	split(in, iv);
	return iv[0];
}

Mat CIQAplatform::circshift(Mat in, const Point& delta) {
	Size sz = in.size();
	assert(sz.height > 0 && sz.width > 0);
	if ((sz.height == 1 && sz.width == 1) || (delta.x == 0 && delta.y == 0))
		return in;
	int x = delta.x;
	int y = delta.y;
	if (x > 0) x = x % sz.width;
	if (y > 0) y = y % sz.height;
	if (x < 0) x = x % sz.width + sz.width;
	if (y < 0) y = y % sz.height + sz.height;
	std::vector<Mat> planes;
	split(in, planes);
	for (size_t i = 0; i < planes.size(); i++) {
		Mat tmp0, tmp1, tmp2, tmp3;
		Mat q0(planes[i], Rect(0, 0, sz.width, sz.height - y));
		Mat q1(planes[i], Rect(0, sz.height - y, sz.width, y));
		q0.copyTo(tmp0);
		q1.copyTo(tmp1);
		tmp0.copyTo(planes[i](Rect(0, y, sz.width, sz.height - y)));
		tmp1.copyTo(planes[i](Rect(0, 0, sz.width, y)));
		Mat q2(planes[i], Rect(0, 0, sz.width - x, sz.height));
		Mat q3(planes[i], Rect(sz.width - x, 0, x, sz.height));
		q2.copyTo(tmp2);
		q3.copyTo(tmp3);
		tmp2.copyTo(planes[i](Rect(x, 0, sz.width - x, sz.height)));
		tmp3.copyTo(planes[i](Rect(0, 0, x, sz.height)));
	}
	Mat out;
	out.create(in.size(), in.type());
	merge(planes, out);
	return out;
}

Mat CIQAplatform::fftshift(Mat in) {
	Size sz = in.size();
	Point pt(0, 0);
	pt.x = (int)floor(sz.width / 2.0);
	pt.y = (int)floor(sz.height / 2.0);
	return circshift(in, pt);
}

Mat CIQAplatform::ifftshift(Mat in) {
	Size sz = in.size();
	Point pt(0, 0);
	pt.x = (int)ceil(sz.width / 2.0);
	pt.y = (int)ceil(sz.height / 2.0);
	return circshift(in, pt);
}

// 视觉对比敏感度函数滤波
Mat CIQAplatform::Make_CSF(Mat in) {
	//CSF构造
	cvtColor(in, in, CV_BGR2GRAY);
	in.convertTo(in, CV_64FC1);
	const int x = in.rows;
	const int y = in.cols;
	const double nfreq = 32.0;
	Mat ICFfrq;
	Mat csf(y, x, CV_64FC1);
	Mat ICFcx(x, y, CV_64FC2);
	Mat gker = (Mat_<double>(7, 7) <<
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0000, 0.0002, 0.0000, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0113, 0.0837, 0.0113, 0.0000, 0.0000,
		0.0000, 0.0002, 0.0837, 0.6187, 0.0837, 0.0002, 0.0000,
		0.0000, 0.0000, 0.0113, 0.0837, 0.0113, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0000, 0.0002, 0.0000, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000);
	Mat xplane(y, x, CV_64FC1);
	Mat yplane(y, x, CV_64FC1);
	Mat xplane0(y, x, CV_64FC1);
	Mat yplane0(y, x, CV_64FC1);
	Mat radfreq(y, x, CV_64FC1);
	Mat angleplane(y, x, CV_64FC1);
	double xb = -x / 2.0 + 0.5, xe = x / 2 - 0.5, yb = -y / 2.0 + 0.5, ye = y / 2.0 - 0.5;
	meshgrid(xplane, yplane, xb, xe, yb, ye);
	xplane = xplane / y * 2 * nfreq;
	yplane = yplane / y * 2 * nfreq;
	xplane0 = xplane.clone();
	yplane0 = yplane.clone();
	multiply(xplane, xplane, xplane);
	multiply(yplane, yplane, yplane);
	for (int i = 0; i < y; i++)
		for (int j = 0; j < x; j++)
			radfreq.ptr<double>(i)[j] = sqrt(xplane.ptr<double>(i)[j] + yplane.ptr<double>(i)[j]); // radial frequency
	double w = 0.7;
	for (int i = 0; i < y; i++)
		for (int j = 0; j < x; j++)
			angleplane.ptr<double>(i)[j] = atan2(yplane0.ptr<double>(i)[j], xplane0.ptr<double>(i)[j]);
	angleplane *= 4;
	for (int i = 0; i < y; i++)
		for (int j = 0; j < x; j++)
			angleplane.ptr<double>(i)[j] = cos(angleplane.ptr<double>(i)[j]);
	angleplane = (1 - w) / 2 * angleplane;
	for (int i = 0; i < y; i++)
		for (int j = 0; j < x; j++)
			angleplane.ptr<double>(i)[j] += (1 + w) / 2;
	divide(radfreq, angleplane, radfreq);
	xplane = radfreq * 0.114;
	yplane = xplane.clone();
	for (int i = 0; i < y; i++)
		for (int j = 0; j < x; j++)
			xplane.ptr<double>(i)[j] += 0.0192;
	xplane *= 2.6;
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			yplane.ptr<double>(i)[j] = pow(yplane.ptr<double>(i)[j], 1.1);
			yplane.ptr<double>(i)[j] = -yplane.ptr<double>(i)[j];
			yplane.ptr<double>(i)[j] = exp(yplane.ptr<double>(i)[j]);
		}
	}
	multiply(xplane, yplane, angleplane);
	std::vector<std::pair<int, int> > p;
	for (int i = 0; i < y; i++)
		for (int j = 0; j < x; j++)
			if (radfreq.ptr<double>(i)[j] < 7.8909)
				p.push_back(std::make_pair(i, j));
	for (auto iter = p.cbegin(); iter != p.cend(); iter++)
		angleplane.ptr<double>(iter->first)[iter->second] = 0.9809;
	csf = angleplane.t();
	for (int i = 0; i < csf.rows; i++)
		for (int j = 0; j < csf.cols; j++)
			csf.ptr<double>(i)[j] = floor(csf.ptr<double>(i)[j] * 10000.0 + 0.5) / 10000.0;

	// CSF滤波（在频域中执行）
	ICFfrq = fft2(in);
	ICFfrq = fftshift(ICFfrq);
	std::vector<Mat> csfv;
	split(ICFfrq, csfv);
	multiply(csfv[0], csf, csfv[0]);
	multiply(csfv[1], csf, csfv[1]);
	merge(csfv, ICFfrq);
	ICFfrq = ifftshift(ICFfrq);
	ICFcx = ifft2(ICFfrq);
	Mat ICF = ireal(ICFcx);
	filter2D(ICF, ICF, CV_64FC1, gker);
	return ICF;
}

// 边缘检测
Mat CIQAplatform::Img_Gradient_Map(Mat in) {
	Mat r_grad_x{}, r_grad_y{}, r_abs_grad_x{}, r_abs_grad_y{};
	Mat gker = (Mat_<double>(9, 9) <<
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0001, 0.0044, 0.0044, 0.0044, 0.0001, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0044, 0.2411, 0.2411, 0.2411, 0.0044, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0044, 0.2411, 0.2411, 0.2411, 0.0044, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0044, 0.2411, 0.2411, 0.2411, 0.0044, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0001, 0.0044, 0.0044, 0.0044, 0.0001, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000);
	Mat IC = in.clone();
	Sobel(IC, r_grad_x, CV_64F, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	Sobel(IC, r_grad_y, CV_64F, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	Mat_<double> IGM(IC.rows, IC.cols);
	for (int i = 0; i < IGM.rows; i++)
		for (int j = 0; j < IGM.cols; j++)
			IGM.ptr<double>(i)[j] = sqrt(r_grad_x.ptr<double>(i)[j] * r_grad_x.ptr<double>(i)[j] + r_grad_y.ptr<double>(i)[j] * r_grad_y.ptr<double>(i)[j]);
	Mat tmpGM, tmpC;
	multiply(IC, IC, tmpC);
	multiply(IGM, IGM, tmpGM);
	Mat_<double> A(IC.rows, IC.cols);
	for (int i = 0; i < A.rows; i++)
		for (int j = 0; j < A.cols; j++)
			A.ptr<double>(i)[j] = (tmpC.ptr<double>(i)[j] + tmpGM.ptr<double>(i)[j]) / 2;
	filter2D(A, A, CV_64FC1, gker);
	for (int i = 0; i < A.rows; i++)
		for (int j = 0; j < A.cols; j++)
			A.ptr<double>(i)[j] = sqrt(A.ptr<double>(i)[j]);
	for (int i = 0; i < A.rows; i++)
		for (int j = 0; j < A.cols; j++)
			if (A.ptr<double>(i)[j] < 25.6)
				A.ptr<double>(i)[j] += 25.6;
	Mat IGMN;
	divide(IGM, A, IGMN);
	return IGMN;
}

void CIQAplatform::Img_Gradient_Map()
{
	// TODO: 在此添加控件通知处理程序代码
	
	Mat in = RefImg.clone();
	Mat IC = Make_CSF(in);
	Mat IGMN = Img_Gradient_Map(IC);
	RIGM = IGMN;
	normalize(IGMN, IGMN, 0.0, 1.0, NORM_MINMAX);
	IGMN.convertTo(IGMN, CV_8UC1, 255.0);
	imshow("RefSal", IGMN);
	/*imwrite("C:\\CSFSaliencyMap_of_ReferenceImg.bmp", IGMN);*/
}

void CIQAplatform::OnBnClickedButtonHc()
{
	// TODO: 在此添加控件通知处理程序代码

	Mat in = RefImg.clone();
	Mat tmp, IHC;
	in.convertTo(tmp, CV_32FC3, 1.0 / 255);
	ImgSRDHC(tmp).convertTo(IHC, CV_8UC1, 255);
	imshow("RefSal", IHC);
	/*imwrite("C:\\HCMap_of_ReferenceImg.bmp", IHC);*/
}


void CIQAplatform::OnBnClickedButtonIg2()
{
	// TODO: 在此添加控件通知处理程序代码

	Mat in = DisImg.clone();
	Mat IC = Make_CSF(in);
	Mat IGMN = Img_Gradient_Map(IC);
	DIGM = IGMN;
	normalize(IGMN, IGMN, 0.0, 1.0, NORM_MINMAX);
	IGMN.convertTo(IGMN, CV_8UC1, 255.0);
	imshow("DSmap", IGMN);
	/*imwrite("C:\\CSFSaliencyMap_of_DistortedImg.bmp", IGMN);*/
}


void CIQAplatform::OnBnClickedButtonHc2()
{
	// TODO: 在此添加控件通知处理程序代码

	Mat in = DisImg.clone();
	Mat tmp, IHC;
	in.convertTo(tmp, CV_32FC3, 1.0 / 255);
	ImgSRDHC(tmp).convertTo(IHC, CV_8UC1, 255);
	imshow("DSmap", IHC);
	/*imwrite("C:\\HCMap_of_DistortedImg.bmp", IHC);*/
}


/* 分块DCT变换 */
void CIQAplatform::BGRBlkDCT(Mat img, R_ARRAY s) {
	// JPEG标准亮度量化矩阵
	double msk[8][8] = { {16,11,10,16,24,40,51,61},
						{12,12,14,19,26,58,60,55},
						{14,13,16,24,40,57,69,56},
						{14,17,22,29,51,87,80,62},
						{18,22,37,56,68,109,103,77},
						{24,35,55,64,81,104,113,92},
						{49,64,78,87,103,121,120,101},
						{72,92,95,98,112,100,103,99} };
	short s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, s8 = 0, s9 = 0;
	int dx = 0, dy = 0;
	double mean = 0.0;
	Mat float_Y, DCTY;
	Mat mask(8, 8, CV_64FC1, msk);
	Rect subwindows;
	img.convertTo(float_Y, CV_64FC1);
	float_Y.copyTo(DCTY);

	// 只变换了亮度分量
	for (int i = 0; i < org.rows / SUB_HEIGHT; i++) {
		for (int j = 0; j < org.cols / SUB_WIDTH; j++) {
			dx = SUB_WIDTH * i;
			dy = SUB_HEIGHT * j;
			subwindows.x = dx;
			subwindows.y = dy;
			subwindows.height = SUB_HEIGHT;
			subwindows.width = SUB_WIDTH;
			dct(float_Y(subwindows), DCTY(subwindows));
			DCTY(subwindows) /= mask;
			s.S0[s0++] = DCTY.ptr<double>(dy)[dx];
			s.S1[s1++] = DCTY.ptr<double>(dy)[dx + 1];
			s.S2[s2++] = DCTY.ptr<double>(dy + 1)[dx];
			s.S3[s3++] = DCTY.ptr<double>(dy + 1)[dx + 1];
			s.S4[s4++] = DCTY.ptr<double>(dy)[dx + 2];
			s.S4[s4++] = DCTY.ptr<double>(dy)[dx + 3];
			s.S4[s4++] = DCTY.ptr<double>(dy + 1)[dx + 2];
			s.S4[s4++] = DCTY.ptr<double>(dy + 1)[dx + 3];
			s.S5[s5++] = DCTY.ptr<double>(dy + 2)[dx];
			s.S5[s5++] = DCTY.ptr<double>(dy + 2)[dx + 1];
			s.S5[s5++] = DCTY.ptr<double>(dy + 3)[dx];
			s.S5[s5++] = DCTY.ptr<double>(dy + 3)[dx + 1];
			s.S6[s6++] = DCTY.ptr<double>(dy + 2)[dx + 2];
			s.S6[s6++] = DCTY.ptr<double>(dy + 2)[dx + 3];
			s.S6[s6++] = DCTY.ptr<double>(dy + 3)[dx + 2];
			s.S6[s6++] = DCTY.ptr<double>(dy + 3)[dx + 3];
			s.S7[s7++] = DCTY.ptr<double>(dy)[dx + 4];
			s.S7[s7++] = DCTY.ptr<double>(dy)[dx + 5];
			s.S7[s7++] = DCTY.ptr<double>(dy)[dx + 6];
			s.S7[s7++] = DCTY.ptr<double>(dy)[dx + 7];
			s.S7[s7++] = DCTY.ptr<double>(dy + 1)[dx + 4];
			s.S7[s7++] = DCTY.ptr<double>(dy + 1)[dx + 5];
			s.S7[s7++] = DCTY.ptr<double>(dy + 1)[dx + 6];
			s.S7[s7++] = DCTY.ptr<double>(dy + 1)[dx + 7];
			s.S7[s7++] = DCTY.ptr<double>(dy + 2)[dx + 4];
			s.S7[s7++] = DCTY.ptr<double>(dy + 2)[dx + 5];
			s.S7[s7++] = DCTY.ptr<double>(dy + 2)[dx + 6];
			s.S7[s7++] = DCTY.ptr<double>(dy + 2)[dx + 7];
			s.S7[s7++] = DCTY.ptr<double>(dy + 3)[dx + 4];
			s.S7[s7++] = DCTY.ptr<double>(dy + 3)[dx + 5];
			s.S7[s7++] = DCTY.ptr<double>(dy + 3)[dx + 6];
			s.S7[s7++] = DCTY.ptr<double>(dy + 3)[dx + 7];
			s.S8[s8++] = DCTY.ptr<double>(dy + 4)[dx];
			s.S8[s8++] = DCTY.ptr<double>(dy + 4)[dx + 1];
			s.S8[s8++] = DCTY.ptr<double>(dy + 4)[dx + 2];
			s.S8[s8++] = DCTY.ptr<double>(dy + 4)[dx + 3];
			s.S8[s8++] = DCTY.ptr<double>(dy + 5)[dx];
			s.S8[s8++] = DCTY.ptr<double>(dy + 5)[dx + 1];
			s.S8[s8++] = DCTY.ptr<double>(dy + 5)[dx + 2];
			s.S8[s8++] = DCTY.ptr<double>(dy + 5)[dx + 3];
			s.S8[s8++] = DCTY.ptr<double>(dy + 6)[dx];
			s.S8[s8++] = DCTY.ptr<double>(dy + 6)[dx + 1];
			s.S8[s8++] = DCTY.ptr<double>(dy + 6)[dx + 2];
			s.S8[s8++] = DCTY.ptr<double>(dy + 6)[dx + 3];
			s.S8[s8++] = DCTY.ptr<double>(dy + 7)[dx];
			s.S8[s8++] = DCTY.ptr<double>(dy + 7)[dx + 1];
			s.S8[s8++] = DCTY.ptr<double>(dy + 7)[dx + 2];
			s.S8[s8++] = DCTY.ptr<double>(dy + 7)[dx + 3];
			s.S9[s9++] = DCTY.ptr<double>(dy + 4)[dx + 4];
			s.S9[s9++] = DCTY.ptr<double>(dy + 4)[dx + 5];
			s.S9[s9++] = DCTY.ptr<double>(dy + 4)[dx + 6];
			s.S9[s9++] = DCTY.ptr<double>(dy + 4)[dx + 7];
			s.S9[s9++] = DCTY.ptr<double>(dy + 5)[dx + 4];
			s.S9[s9++] = DCTY.ptr<double>(dy + 5)[dx + 5];
			s.S9[s9++] = DCTY.ptr<double>(dy + 5)[dx + 6];
			s.S9[s9++] = DCTY.ptr<double>(dy + 5)[dx + 7];
			s.S9[s9++] = DCTY.ptr<double>(dy + 6)[dx + 4];
			s.S9[s9++] = DCTY.ptr<double>(dy + 6)[dx + 5];
			s.S9[s9++] = DCTY.ptr<double>(dy + 6)[dx + 6];
			s.S9[s9++] = DCTY.ptr<double>(dy + 6)[dx + 7];
			s.S9[s9++] = DCTY.ptr<double>(dy + 7)[dx + 4];
			s.S9[s9++] = DCTY.ptr<double>(dy + 7)[dx + 5];
			s.S9[s9++] = DCTY.ptr<double>(dy + 7)[dx + 6];
			s.S9[s9++] = DCTY.ptr<double>(dy + 7)[dx + 7];
		}
	}
}

std::vector<Mat> CIQAplatform::dct10subbands(R_ARRAY RDCTArray, Mat RDCT, Mat RDCT1) {
	short s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, s8 = 0, s9 = 0;
	std::vector<Mat> ds;
	Mat T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, tmp2, tmp3;
	T0.create(Size(NORM_WIDTH / SUB_WIDTH, NORM_HEIGHT / SUB_HEIGHT), CV_32FC1);
	T1.create(Size(NORM_WIDTH / SUB_WIDTH, NORM_HEIGHT / SUB_HEIGHT), CV_32FC1);
	T2.create(Size(NORM_WIDTH / SUB_WIDTH, NORM_HEIGHT / SUB_HEIGHT), CV_32FC1);
	T3.create(Size(NORM_WIDTH / SUB_WIDTH, NORM_HEIGHT / SUB_HEIGHT), CV_32FC1);
	T4.create(Size(2 * (NORM_WIDTH / SUB_WIDTH), 2 * (NORM_HEIGHT / SUB_HEIGHT)), CV_32FC1);
	T5.create(Size(2 * (NORM_WIDTH / SUB_WIDTH), 2 * (NORM_HEIGHT / SUB_HEIGHT)), CV_32FC1);
	T6.create(Size(2 * (NORM_WIDTH / SUB_WIDTH), 2 * (NORM_HEIGHT / SUB_HEIGHT)), CV_32FC1);
	T7.create(Size(4 * (NORM_WIDTH / SUB_WIDTH), 4 * (NORM_WIDTH / SUB_WIDTH)), CV_32FC1);
	T8.create(Size(4 * (NORM_WIDTH / SUB_WIDTH), 4 * (NORM_WIDTH / SUB_WIDTH)), CV_32FC1);
	T9.create(Size(4 * (NORM_WIDTH / SUB_WIDTH), 4 * (NORM_WIDTH / SUB_WIDTH)), CV_32FC1);
	tmp2.create(Size(NORM_WIDTH / SUB_WIDTH, NORM_HEIGHT / SUB_HEIGHT), CV_16UC1);
	for (int i = 0; i < NORM_WIDTH / SUB_WIDTH; i++) {
		for (int j = 0; j < NORM_HEIGHT / SUB_HEIGHT; j++) {
			RDCT.ptr<double>(j)[i] = RDCTArray.S0[s0++];
			T0.ptr<float>(j)[i] = RDCT.ptr<double>(j)[i];
			tmp2.ptr<short>(j)[i] = 32 * RDCT.ptr<double>(j)[i];
		}
	}
	ds.push_back(T0.clone());
	T0.convertTo(I0, CV_8UC1, 1.0 / 32);

	for (int i = NORM_WIDTH / SUB_WIDTH; i < 2 * (NORM_WIDTH / SUB_WIDTH); i++) {
		for (int j = 0; j < NORM_HEIGHT / SUB_HEIGHT; j++) {
			RDCT.ptr<double>(j)[i] = RDCTArray.S1[s1++];
			T1.ptr<float>(j)[i - NORM_WIDTH / SUB_WIDTH] = RDCT.ptr<double>(j)[i];
		}
	}
	T1.convertTo(I1, CV_8UC1);
	ds.push_back(T1.clone());
	for (int i = 0; i < NORM_WIDTH / SUB_WIDTH; i++) {
		for (int j = NORM_HEIGHT / SUB_HEIGHT; j < 2 * (NORM_HEIGHT / SUB_HEIGHT); j++) {
			RDCT.ptr<double>(j)[i] = RDCTArray.S2[s2++];
			T2.ptr<float>(j - NORM_HEIGHT / SUB_HEIGHT)[i] = RDCT.ptr<double>(j)[i];
		}
	}
	T2.convertTo(I2, CV_8UC1);
	ds.push_back(T2.clone());
	for (int i = NORM_WIDTH / SUB_WIDTH; i < 2 * (NORM_WIDTH / SUB_WIDTH); i++) {
		for (int j = NORM_HEIGHT / SUB_HEIGHT; j < 2 * (NORM_HEIGHT / SUB_HEIGHT); j++) {
			RDCT.ptr<double>(j)[i] = RDCTArray.S3[s3++];
			T3.ptr<float>(j - NORM_HEIGHT / SUB_HEIGHT)[i - NORM_HEIGHT / SUB_HEIGHT] = RDCT.ptr<double>(j)[i];
		}
	}
	T3.convertTo(I3, CV_8UC1);
	ds.push_back(T3.clone());
	for (int i = 2 * (NORM_HEIGHT / SUB_HEIGHT); i < 4 * (NORM_WIDTH / SUB_WIDTH); i += 2) {
		for (int j = 0; j < 2 * (NORM_HEIGHT / SUB_HEIGHT); j += 2) {
			RDCT.ptr<double>(j)[i] = RDCTArray.S4[s4++];
			RDCT.ptr<double>(j)[i + 1] = RDCTArray.S4[s4++];
			RDCT.ptr<double>(j + 1)[i] = RDCTArray.S4[s4++];
			RDCT.ptr<double>(j + 1)[i + 1] = RDCTArray.S4[s4++];
			T4.ptr<float>(j)[i - 2 * (NORM_HEIGHT / SUB_HEIGHT)] = RDCT.ptr<double>(j)[i];
			T4.ptr<float>(j)[i + 1 - 2 * (NORM_HEIGHT / SUB_HEIGHT)] = RDCT.ptr<double>(j)[i + 1];
			T4.ptr<float>(j + 1)[i - 2 * (NORM_HEIGHT / SUB_HEIGHT)] = RDCT.ptr<double>(j + 1)[i];
			T4.ptr<float>(j + 1)[i + 1 - 2 * (NORM_HEIGHT / SUB_HEIGHT)] = RDCT.ptr<double>(j + 1)[i + 1];
		}
	}
	T4.convertTo(I4, CV_8UC1);
	ds.push_back(T4.clone());
	for (int i = 0; i < 2 * (NORM_WIDTH / SUB_WIDTH); i += 2) {
		for (int j = 2 * (NORM_HEIGHT / SUB_HEIGHT); j < 4 * (NORM_HEIGHT / SUB_HEIGHT); j += 2) {
			RDCT.ptr<double>(j)[i] = RDCTArray.S5[s5++];
			RDCT.ptr<double>(j)[i + 1] = RDCTArray.S5[s5++];
			RDCT.ptr<double>(j + 1)[i] = RDCTArray.S5[s5++];
			RDCT.ptr<double>(j + 1)[i + 1] = RDCTArray.S5[s5++];
			T5.ptr<float>(j - 2 * (NORM_HEIGHT / SUB_HEIGHT))[i] = RDCT.ptr<double>(j)[i];
			T5.ptr<float>(j - 2 * (NORM_HEIGHT / SUB_HEIGHT))[i + 1] = RDCT.ptr<double>(j)[i + 1];
			T5.ptr<float>(j + 1 - 2 * (NORM_HEIGHT / SUB_HEIGHT))[i] = RDCT.ptr<double>(j + 1)[i];
			T5.ptr<float>(j + 1 - 2 * (NORM_HEIGHT / SUB_HEIGHT))[i + 1] = RDCT.ptr<double>(j + 1)[i + 1];
		}
	}
	T5.convertTo(I5, CV_8UC1);
	ds.push_back(T5.clone());
	for (int i = 2 * (NORM_HEIGHT / SUB_HEIGHT); i < 4 * (NORM_WIDTH / SUB_WIDTH); i += 2) {
		for (int j = 2 * (NORM_HEIGHT / SUB_HEIGHT); j < 4 * (NORM_HEIGHT / SUB_HEIGHT); j += 2) {
			RDCT.ptr<double>(j)[i] = RDCTArray.S6[s6++];
			RDCT.ptr<double>(j)[i + 1] = RDCTArray.S6[s6++];
			RDCT.ptr<double>(j + 1)[i] = RDCTArray.S6[s6++];
			RDCT.ptr<double>(j + 1)[i + 1] = RDCTArray.S6[s6++];
			T6.ptr<float>(j - 2 * (NORM_HEIGHT / SUB_HEIGHT))[i - 2 * (NORM_HEIGHT / SUB_HEIGHT)] = RDCT.ptr<double>(j)[i];
			T6.ptr<float>(j - 2 * (NORM_HEIGHT / SUB_HEIGHT))[i + 1 - 2 * (NORM_HEIGHT / SUB_HEIGHT)] = RDCT.ptr<double>(j)[i + 1];
			T6.ptr<float>(j + 1 - 2 * (NORM_HEIGHT / SUB_HEIGHT))[i - 2 * (NORM_HEIGHT / SUB_HEIGHT)] = RDCT.ptr<double>(j + 1)[i];
			T6.ptr<float>(j + 1 - 2 * (NORM_HEIGHT / SUB_HEIGHT))[i + 1 - 2 * (NORM_HEIGHT / SUB_HEIGHT)] = RDCT.ptr<double>(j + 1)[i + 1];
		}
	}
	T6.convertTo(I6, CV_8UC1);
	ds.push_back(T6.clone());
	for (int i = 4 * (NORM_WIDTH / SUB_WIDTH); i < 8 * (NORM_WIDTH / SUB_WIDTH); i += 4) {
		for (int j = 0; j < 4 * (NORM_HEIGHT / SUB_HEIGHT); j += 4) {
			RDCT.ptr<double>(j)[i] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j)[i + 1] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j)[i + 2] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j)[i + 3] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j + 1)[i] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j + 1)[i + 1] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j + 1)[i + 2] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j + 1)[i + 3] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j + 2)[i] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j + 2)[i + 1] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j + 2)[i + 2] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j + 2)[i + 3] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j + 3)[i] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j + 3)[i + 1] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j + 3)[i + 2] = RDCTArray.S7[s7++];
			RDCT.ptr<double>(j + 3)[i + 3] = RDCTArray.S7[s7++];
			T7.ptr<float>(j)[i - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j)[i];
			T7.ptr<float>(j)[i + 1 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j)[i + 1];
			T7.ptr<float>(j)[i + 2 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j)[i + 2];
			T7.ptr<float>(j)[i + 3 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j)[i + 3];
			T7.ptr<float>(j + 1)[i - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 1)[i];
			T7.ptr<float>(j + 1)[i + 1 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 1)[i + 1];
			T7.ptr<float>(j + 1)[i + 2 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 1)[i + 2];
			T7.ptr<float>(j + 1)[i + 3 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 1)[i + 3];
			T7.ptr<float>(j + 2)[i - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 2)[i];
			T7.ptr<float>(j + 2)[i + 1 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 2)[i + 1];
			T7.ptr<float>(j + 2)[i + 2 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 2)[i + 2];
			T7.ptr<float>(j + 2)[i + 3 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 2)[i + 3];
			T7.ptr<float>(j + 3)[i - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 3)[i];
			T7.ptr<float>(j + 3)[i + 1 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 3)[i + 1];
			T7.ptr<float>(j + 3)[i + 2 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 3)[i + 2];
			T7.ptr<float>(j + 3)[i + 3 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 3)[i + 3];
		}
	}
	T7.convertTo(I7, CV_8UC1);
	ds.push_back(T7.clone());
	for (int i = 0; i < 4 * (NORM_WIDTH / SUB_WIDTH); i += 4) {
		for (int j = 4 * (NORM_HEIGHT / SUB_HEIGHT); j < 8 * (NORM_HEIGHT / SUB_HEIGHT); j += 4) {
			RDCT.ptr<double>(j)[i] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j)[i + 1] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j)[i + 2] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j)[i + 3] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j + 1)[i] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j + 1)[i + 1] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j + 1)[i + 2] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j + 1)[i + 3] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j + 2)[i] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j + 2)[i + 1] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j + 2)[i + 2] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j + 2)[i + 3] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j + 3)[i] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j + 3)[i + 1] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j + 3)[i + 2] = RDCTArray.S8[s8++];
			RDCT.ptr<double>(j + 3)[i + 3] = RDCTArray.S8[s8++];
			T8.ptr<float>(j - 4 * (NORM_WIDTH / SUB_WIDTH))[i] = RDCT.ptr<double>(j)[i];
			T8.ptr<float>(j - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 1] = RDCT.ptr<double>(j)[i + 1];
			T8.ptr<float>(j - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 2] = RDCT.ptr<double>(j)[i + 2];
			T8.ptr<float>(j - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 3] = RDCT.ptr<double>(j)[i + 3];
			T8.ptr<float>(j + 1 - 4 * (NORM_WIDTH / SUB_WIDTH))[i] = RDCT.ptr<double>(j + 1)[i];
			T8.ptr<float>(j + 1 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 1] = RDCT.ptr<double>(j + 1)[i + 1];
			T8.ptr<float>(j + 1 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 2] = RDCT.ptr<double>(j + 1)[i + 2];
			T8.ptr<float>(j + 1 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 3] = RDCT.ptr<double>(j + 1)[i + 3];
			T8.ptr<float>(j + 2 - 4 * (NORM_WIDTH / SUB_WIDTH))[i] = RDCT.ptr<double>(j + 2)[i];
			T8.ptr<float>(j + 2 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 1] = RDCT.ptr<double>(j + 2)[i + 1];
			T8.ptr<float>(j + 2 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 2] = RDCT.ptr<double>(j + 2)[i + 2];
			T8.ptr<float>(j + 2 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 3] = RDCT.ptr<double>(j + 2)[i + 3];
			T8.ptr<float>(j + 3 - 4 * (NORM_WIDTH / SUB_WIDTH))[i] = RDCT.ptr<double>(j + 3)[i];
			T8.ptr<float>(j + 3 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 1] = RDCT.ptr<double>(j + 3)[i + 1];
			T8.ptr<float>(j + 3 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 2] = RDCT.ptr<double>(j + 3)[i + 2];
			T8.ptr<float>(j + 3 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 3] = RDCT.ptr<double>(j + 3)[i + 3];
		}
	}
	T8.convertTo(I8, CV_8UC1);
	ds.push_back(T8.clone());
	for (int i = 4 * (NORM_HEIGHT / SUB_HEIGHT); i < 8 * (NORM_WIDTH / SUB_WIDTH); i += 4) {
		for (int j = 4 * (NORM_HEIGHT / SUB_HEIGHT); j < 8 * (NORM_HEIGHT / SUB_HEIGHT); j += 4) {
			RDCT.ptr<double>(j)[i] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j)[i + 1] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j)[i + 2] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j)[i + 3] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j + 1)[i] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j + 1)[i + 1] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j + 1)[i + 2] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j + 1)[i + 3] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j + 2)[i] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j + 2)[i + 1] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j + 2)[i + 2] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j + 2)[i + 3] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j + 3)[i] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j + 3)[i + 1] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j + 3)[i + 2] = RDCTArray.S9[s9++];
			RDCT.ptr<double>(j + 3)[i + 3] = RDCTArray.S9[s9++];
			T9.ptr<float>(j - 4 * (NORM_WIDTH / SUB_WIDTH))[i - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j)[i];
			T9.ptr<float>(j - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 1 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j)[i + 1];
			T9.ptr<float>(j - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 2 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j)[i + 2];
			T9.ptr<float>(j - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 3 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j)[i + 3];
			T9.ptr<float>(j + 1 - 4 * (NORM_WIDTH / SUB_WIDTH))[i - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 1)[i];
			T9.ptr<float>(j + 1 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 1 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 1)[i + 1];
			T9.ptr<float>(j + 1 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 2 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 1)[i + 2];
			T9.ptr<float>(j + 1 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 3 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 1)[i + 3];
			T9.ptr<float>(j + 2 - 4 * (NORM_WIDTH / SUB_WIDTH))[i - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 2)[i];
			T9.ptr<float>(j + 2 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 1 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 2)[i + 1];
			T9.ptr<float>(j + 2 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 2 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 2)[i + 2];
			T9.ptr<float>(j + 2 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 3 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 2)[i + 3];
			T9.ptr<float>(j + 3 - 4 * (NORM_WIDTH / SUB_WIDTH))[i - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 3)[i];
			T9.ptr<float>(j + 3 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 1 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 3)[i + 1];
			T9.ptr<float>(j + 3 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 2 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 3)[i + 2];
			T9.ptr<float>(j + 3 - 4 * (NORM_WIDTH / SUB_WIDTH))[i + 3 - 4 * (NORM_WIDTH / SUB_WIDTH)] = RDCT.ptr<double>(j + 3)[i + 3];
		}
	}
	T9.convertTo(I9, CV_8UC1);
	ds.push_back(T9.clone());
	RDCT.convertTo(RDCT1, CV_8UC1);
	tmp2.convertTo(tmp3, CV_8UC1, 1.0 / 255.0);
	for (int i = 0; i < NORM_HEIGHT; i++)
		for (int j = 0; j < NORM_WIDTH; j++)
			RDCT1.ptr<uchar>(i)[j] = 255 - /*5 * */RDCT1.ptr<uchar>(i)[j];
	for (int i = 0; i < NORM_WIDTH / SUB_WIDTH; i++)
		for (int j = 0; j < NORM_HEIGHT / SUB_HEIGHT; j++)
			RDCT1.ptr<uchar>(i)[j] = tmp3.ptr<uchar>(i)[j];
	ds.push_back(RDCT.clone());
	ds.push_back(RDCT1.clone());
	return ds;
}

double CIQAplatform::calc3orderMom(Mat& channel)
{
	uchar* p;
	double mom = 0;
	double m = mean(channel)[0];
	int nRows = channel.rows;
	int nCols = channel.cols;
	if (channel.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}
	for (int i = 0; i < nRows; i++)
	{
		p = channel.ptr<uchar>(i);
		for (int j = 0; j < nCols; j++)
			mom += (p[j] - m) * (p[j] - m) * (p[j] - m);
	}
	float temp;
	temp = cvCbrt((float)(mom / (nRows * nCols)));
	mom = (double)temp;
	return mom;
}

/* 计算9个颜色矩：3个通道的1、2、3阶矩 */
void CIQAplatform::colorMom(Mat & img, double* Mom) {
	Mat r(img.rows, img.cols, CV_8U);
	Mat g(img.rows, img.cols, CV_8U);
	Mat b(img.rows, img.cols, CV_8U);
	Mat channels[] = { b, g, r };
	split(img, channels);
	Mat tmp_m, tmp_sd;

	meanStdDev(b, tmp_m, tmp_sd);
	Mom[0] = tmp_m.at<double>(0, 0);
	Mom[3] = tmp_sd.at<double>(0, 0);
	Mom[6] = calc3orderMom(b);

	meanStdDev(g, tmp_m, tmp_sd);
	Mom[1] = tmp_m.at<double>(0, 0);
	Mom[4] = tmp_sd.at<double>(0, 0);
	Mom[7] = calc3orderMom(g);

	meanStdDev(r, tmp_m, tmp_sd);
	Mom[2] = tmp_m.at<double>(0, 0);
	Mom[5] = tmp_sd.at<double>(0, 0);
	Mom[8] = calc3orderMom(r);
}

CString CIQAplatform::ColorMomentRef(Mat Img, double* Mom1) {
	colorMom(Img, Mom1);
	CString str1 = _T(""), ts1 = _T(""), str2 = _T(""), ts2 = _T("");
	for (int i = 0; i < 9; i++) {
		ts1.Format(_T("-%lf"), Mom1[i]);
		str1 += ts1;
	}
	return str1;
}

double* CIQAplatform::TenengradMeasure(double* score, Mat image) {

	/*GaussianBlur(image, image, Size(3, 3), 0, 0, BORDER_DEFAULT);*/

	Mat ImgGray = Mat::zeros(image.size(), image.type());
	cvtColor(image, ImgGray, CV_BGR2GRAY);
	Mat grad_x{}, grad_y{};
	Mat abs_grad_x{}, abs_grad_y{}, abs_grad_x2{}, abs_grad_y2{};
	Sobel(ImgGray, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(ImgGray, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	pow(abs_grad_x, 2, abs_grad_x2);
	pow(abs_grad_y, 2, abs_grad_y2);
	Mat imgSobel = Mat::zeros(image.size(), CV_8U);
	addWeighted(abs_grad_x2, 1, abs_grad_y2, 1, 0, imgSobel);

	double grad_value = 0;
	for (int i = 0; i < imgSobel.rows; i++) {
		uchar* data = imgSobel.ptr<uchar>(i);
		for (int j = 0; j < imgSobel.cols; j++) {
			grad_value += data[j];
		}
	}
	*score = grad_value / (imgSobel.rows) / (imgSobel.cols);
	return score;
}

double CIQAplatform::NRSSMeasure(Mat img) {
	Mat image = img;
	assert(image.empty());
	Mat gray_img, Ir, G, Gr;
	if (image.channels() == 3) {
		cvtColor(image, gray_img, CV_BGR2GRAY);
	}

	//构造参考图像	
	GaussianBlur(gray_img, Ir, cv::Size(7, 7), 6, 6);

	//提取图像和参考图像的梯度信息	
	Sobel(gray_img, G, CV_32FC1, 1, 1);

	//计算原始图像sobel梯度	
	Sobel(Ir, Gr, CV_32FC1, 1, 1);

	//计算构造函数的sobel梯度 	
	//找出梯度图像 G 中梯度信息最丰富的 N 个图像块，n=64(即划分为8x8的大小)	
	//计算每个小方块的宽/高	
	int block_cols = G.cols * 2 / 9;
	int block_rows = G.rows * 2 / 9;

	//获取方差最大的block	

	Mat best_G, best_Gr;
	float max_stddev = .0f;
	int pos = 0;
	for (int i = 0; i < 64; ++i) {
		int left_x = (i % 8) * (block_cols / 2);
		int left_y = (i / 8) * (block_rows / 2);
		int right_x = left_x + block_cols;
		int right_y = left_y + block_rows;
		if (left_x < 0)
			left_x = 0;
		if (left_y < 0)
			left_y = 0;
		if (right_x >= G.cols)
			right_x = G.cols - 1;
		if (right_y >= G.rows)
			right_y = G.rows - 1;
		Rect roi(left_x, left_y, right_x - left_x, right_y - left_y);
		Mat temp = G(roi).clone();
		Scalar mean, stddev;
		meanStdDev(temp, mean, stddev);
		if (stddev.val[0] > max_stddev) {
			max_stddev = static_cast<float>(stddev.val[0]);
			pos = i;
			best_G = temp;
			best_Gr = Gr(roi).clone();
		}
	}

	const double C1 = 6.5025, C2 = 58.5225;
	int d = CV_32F;
	Mat I1, I2;
	best_G.convertTo(I1, d);
	best_Gr.convertTo(I2, d);
	Mat I1_2 = I1.mul(I1);
	Mat I2_2 = I2.mul(I2);
	Mat I1_I2 = I1.mul(I2);
	Mat mu1, mu2;
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);
	Mat sigma1_2, sigam2_2, sigam12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigam2_2, Size(11, 11), 1.5);
	sigam2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigam12, Size(11, 11), 1.5);
	sigam12 -= mu1_mu2;
	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigam12 + C2;
	t3 = t1.mul(t2);
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigam2_2 + C2;
	t1 = t1.mul(t2);
	Mat ssim_map;
	divide(t3, t1, ssim_map);
	Scalar mssim = mean(ssim_map);

	return 1 - (mssim.val[0] + mssim.val[1] + mssim.val[2]) / 3;
}

#pragma optimize("", off)

void CIQAplatform::OnRRGSM()
{
	// TODO: 在此添加控件通知处理程序代码
	/* 创建进度条 */
	CProcDlg* pTest = new CProcDlg();
	pTest->Create(IDD_DIALOG1);
	pIQA m_RR = (pIQA)malloc(sizeof(_IQAPCDlg));
	m_RR->cpd = pTest;
	m_RR->cia = this;
	pTest->ShowWindow(SW_SHOW);

	/* 开始模型测试 */
	AfxBeginThread(Compute, m_RR, THREAD_PRIORITY_NORMAL);
}



UINT Compute(LPVOID lpParam) {
	double* Mom1 = (double*)malloc(9 * sizeof(double));
	double* Mom2 = (double*)malloc(9 * sizeof(double));
	double* ap = (double*)malloc(sizeof(double));
	double* ad = (double*)malloc(sizeof(double));
	Mat RDCT, RDCT1, DisRDCT, DisRDCT1, RefSal, DisSal, R_W, D_W, RGm, DGm;
	R_ARRAY RDCTArray, DisRDCTArray;
	RDCT.create(Size(NORM_WIDTH, NORM_HEIGHT), CV_64FC1);
	RDCT1.create(Size(NORM_WIDTH, NORM_HEIGHT), CV_8UC1);
	DisRDCT.create(Size(NORM_WIDTH, NORM_HEIGHT), CV_64FC1);
	DisRDCT1.create(Size(NORM_WIDTH, NORM_HEIGHT), CV_8UC1);
	R_W.create(Size(NORM_WIDTH, NORM_HEIGHT), CV_8UC1);
	D_W.create(Size(NORM_WIDTH, NORM_HEIGHT), CV_8UC1);
	RGm.create(Size(NORM_WIDTH, NORM_HEIGHT), CV_8UC1);
	DGm.create(Size(NORM_WIDTH, NORM_HEIGHT), CV_8UC1);


	USES_CONVERSION;
	// 更改DBS的值为当前使用的图像数据库；更改data_path为实际的数据集路径
	/*************************************/
	IMG_DBS DBS = TID2008;
	const CString data_path = _T("F:\\dataset\\");
	/*************************************/

	// 不需要更改此处
	int refimg_num = 25, disimg_num = 120;
	CStdioFile FA, MOS, RD, DD;
	CString mostr = _T(""), im1 = _T(""), im2 = _T(""), P = _T(""), P1 = _T(""), P2 = _T(""), P3 = _T(""), m_strFilePathDst = _T("");
	std::vector<CString> mos, mse, psnr, psnrhvs, wsnr, nqm, ssim, mssim, vif, vifp, dcv, ddv;

	switch (DBS) {
	case TID2013:
		refimg_num = 25;
		disimg_num = 120;
		MOS.Open(data_path + _T("tid2013\\mos.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			mos.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2013\\metrics_values\\PSNR.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			psnr.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2013\\metrics_values\\PSNRHVS.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			psnrhvs.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2013\\metrics_values\\WSNR.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			wsnr.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2013\\metrics_values\\NQM.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			nqm.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2013\\metrics_values\\SSIM.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			ssim.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2013\\metrics_values\\MSSIM.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			mssim.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2013\\metrics_values\\VIFP.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			vifp.push_back(mostr);
		MOS.Close();
		FA.Open(data_path + _T("tid2013\\tid20131.txt"), CFile::modeCreate | CFile::modeReadWrite);
		break;

	case TID2008:
		refimg_num = 25;
		disimg_num = 68;
		MOS.Open(data_path + _T("tid2008\\mos.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			mos.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2008\\metrics_values\\mse.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			mse.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2008\\metrics_values\\psnr.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			psnr.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2008\\metrics_values\\wsnr.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			wsnr.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2008\\metrics_values\\nqm.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			nqm.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2008\\metrics_values\\ssim.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			ssim.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2008\\metrics_values\\MSSIM.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			mssim.push_back(mostr);
		MOS.Close();
		MOS.Open(data_path + _T("tid2008\\metrics_values\\vif.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			vif.push_back(mostr);
		MOS.Close();
		FA.Open(data_path + _T("tid2008\\tid20081.txt"), CFile::modeCreate | CFile::modeReadWrite);
		break;

	case LIVE2:

		break;

	case CSIQ:
		refimg_num = 30;
		disimg_num = 5;
		MOS.Open(data_path + _T("CSIQ\\distorted\\contrastre\\dmos.txt"), CFile::modeRead);
		while (MOS.ReadString(mostr))
			mos.push_back(mostr);
		MOS.Close();
		FA.Open(data_path + _T("CSIQ\\distorted\\contrastre\\contrast1.txt"), CFile::modeCreate | CFile::modeReadWrite);
		break;

	case USCSIPI:

		break;

	case UCID:

		break;

	case koniq_10k:

		break;

	case CLIVE:

		break;

	default:

		break;
	}

	int progress = 0;
	pIQA pTest = (pIQA)lpParam;
	CProcDlg * pDlg = pTest->cpd;
	((CProgressCtrl*)pDlg->GetDlgItem(IDC_PROGRESS1))->SetRange(0, (short)mos.size() - 1);
	struct _finddata_t fileinfo;
	intptr_t h2 = NULL;
	std::string tstr = "";

	for (int j = 0; j < refimg_num; j++) {
		switch (DBS) {
		case TID2013: {
			P1 = data_path + _T("tid2013\\reference_images\\");
			P2 = data_path + _T("tid2013\\distorted_images\\");
			P = P2;
			if (j < 9) {
				im1.Format(_T("I0%d.BMP"), j + 1);
				im2.Format(_T("i0%d*.bmp"), j + 1);
			}
			else {
				im1.Format(_T("I%d.BMP"), j + 1);
				im2.Format(_T("i%d*.bmp"), j + 1);
			}
			P1 += im1;
			P2 += im2;
			tstr = W2A(P2.AllocSysString());
			const char* t3p2 = tstr.c_str();
			h2 = _findfirst(t3p2, &fileinfo);
			break;
		}
		case TID2008: {
			P1 = data_path + _T("tid2008\\reference_images\\");
			P2 = data_path + _T("tid2008\\distorted_images\\");
			P = P2;
			if (j < 9) {
				im1.Format(_T("I0%d.BMP"), j + 1);
				im2.Format(_T("i0%d*.bmp"), j + 1);
			}
			else {
				im1.Format(_T("I%d.BMP"), j + 1);
				im2.Format(_T("i%d*.bmp"), j + 1);
			}
			P1 += im1;
			P2 += im2;
			tstr = W2A(P2.AllocSysString());
			const char* t8p2 = tstr.c_str();
			h2 = _findfirst(t8p2, &fileinfo);
			break;
		}
		case LIVE2: {

			break;
		}
		case CSIQ: {
			P1 = data_path + _T("CSIQ\\reference\\");
			P2 = data_path + _T("CSIQ\\distorted\\contrastre\\");
			P = P2;
			if (j < 9) {
				im1.Format(_T("I0%d.png"), j + 1);
				im2.Format(_T("I0%d*.png"), j + 1);
			}
			else {
				im1.Format(_T("I%d.png"), j + 1);
				im2.Format(_T("I%d*.png"), j + 1);
			}
			P1 += im1;
			P2 += im2;
			tstr = W2A(P2.AllocSysString());
			const char* cp2 = tstr.c_str();
			h2 = _findfirst(cp2, &fileinfo);
			break;
		}
		}

		// 预测数据集中所有图像的质量评分
		do {
			((CProgressCtrl*)pDlg->GetDlgItem(IDC_PROGRESS1))->SetPos(progress);
			std::string refstr = "", disstr = "";
			CString st, r = _T(""), d = _T(""), strb = _T(""), strg = _T(""), strr = _T(""), dstrb = _T(""), dstrg = _T(""), dstrr = _T("");
			CString dts(((std::string)fileinfo.name).c_str());
			P3 = P;
			P3 += dts;

			RefImg = imread(W2A(P1.AllocSysString()));
			DisImg = imread(W2A(P3.AllocSysString()));

			ap = pTest->cia->TenengradMeasure(ap, RefImg);
			ad = pTest->cia->TenengradMeasure(ad, DisImg);
			r = pTest->cia->ColorMomentRef(RefImg, Mom1);
			d = pTest->cia->ColorMomentRef(DisImg, Mom2);

			resize(RefImg, org, Size(NORM_WIDTH, NORM_HEIGHT), 0, 0, INTER_LINEAR);
			resize(DisImg, dis, Size(NORM_WIDTH, NORM_HEIGHT), 0, 0, INTER_LINEAR);
			RefSal = pTest->cia->Make_CSF(org);
			DisSal = pTest->cia->Make_CSF(dis);
			RGm = pTest->cia->Img_Gradient_Map(RefSal);
			DGm = pTest->cia->Img_Gradient_Map(DisSal);
			RGm.convertTo(R_W, CV_8UC1, 255.0);
			DGm.convertTo(D_W, CV_8UC1, 255.0);
			pTest->cia->BGRBlkDCT(R_W, RDCTArray);
			pTest->cia->BGRBlkDCT(D_W, DisRDCTArray);
			std::vector<Mat> rd = pTest->cia->dct10subbands(RDCTArray, RDCT, RDCT1);
			std::vector<Mat> dd = pTest->cia->dct10subbands(DisRDCTArray, DisRDCT, DisRDCT1);
			double rtrpy[10] = { 0.0 };
			double dtrpy[10] = { 0.0 };
			/* Entropy of each Subband */
			for (int k = 0; k < 10; k++) {
				double rntrpy = 0.0, dntrpy = 0.0;
				double rhist[256] = { 0.0 };
				double dhist[256] = { 0.0 };
				Mat Rt, Dt, RDW, DDW;
				normalize(rd[k], Rt, 0.0, 256.0, NORM_MINMAX);
				normalize(dd[k], Dt, 0.0, 256.0, NORM_MINMAX);
				Rt.convertTo(RDW, CV_16UC1);
				Dt.convertTo(DDW, CV_16UC1);
				for (int i = 0; i < DDW.rows; i++) {
					const short* data1 = RDW.ptr<short>(i);
					const short* data2 = DDW.ptr<short>(i);
					for (int j = 0; j < DDW.cols; j++) {
						rhist[data1[j]]++;
						dhist[data2[j]]++;
					}
				}
				for (int i = 0; i < 256; i++) {
					rhist[i] /= (RDW.rows * RDW.cols);
					dhist[i] /= (DDW.rows * DDW.cols);
				}
				for (int i = 0; i < 256; i++) {
					rntrpy += (-rhist[i] * (log2(1 + rhist[i])));
					dntrpy += (-dhist[i] * (log2(1 + dhist[i])));
				}
				rtrpy[k] = rntrpy;
				dtrpy[k] = dntrpy;
			}


			/* score */
			double a = 0.05 * sqrt(ABS(*ap - *ad));
			for (int j = 0; j < 9; j++)
				Mom1[j] = (2 * Mom1[j] * Mom2[j]) / (Mom1[j] * Mom1[j] + Mom2[j] * Mom2[j]);
			double c = 1 / exp(sqrt(sqrt(ABS(1 - (Mom1[0] + Mom1[1] + Mom1[2] + Mom1[3] + Mom1[4] + Mom1[5] + Mom1[6] + Mom1[7] + Mom1[8]) / 9))));
			double b = 0.0;
			/*for (int i = 0; i < 10; i++)
				b += ABS(sqrt(ABS(rtrpy[i])) - sqrt(ABS(dtrpy[i])));
			b *= 5;*/
			for (int i = 0; i < 10; i++)
				b += ABS(rtrpy[i] - dtrpy[i]);
			b = exp(-b);
			/*double score = 9 / (1 + log2(1 + a + b)) * c;*/
			double score = 9 / (1 + log2(1 + a)) * b * c;
			strb.Format(_T("%.5lf"), score);
			CString fn(((std::string)fileinfo.name).c_str());

			switch (DBS) {
			case TID2013:
				st = _T("");
				st += fn;
				st += " ";
				st += strb;
				st += " ";
				st += mos[progress];
				/*st += " ";
				st += mse[progress];*/
				st += " ";
				st += psnrhvs[progress];
				st += " ";
				st += psnr[progress];
				st += " ";
				st += wsnr[progress];
				st += " ";
				st += nqm[progress];
				st += " ";
				st += ssim[progress];
				st += " ";
				st += mssim[progress];
				/*st += " ";
				st += vif[progress++];*/
				st += " ";
				st += vifp[progress++];
				break;

			case TID2008:
				st = _T("");
				st += fn;
				st += " ";
				st += strb;
				st += " ";
				st += mos[progress];
				st += " ";
				st += mse[progress];
				/*st += " ";
				st += psnrhvs[progress];*/
				st += " ";
				st += psnr[progress];
				st += " ";
				st += wsnr[progress];
				st += " ";
				st += nqm[progress];
				st += " ";
				st += ssim[progress];
				st += " ";
				st += mssim[progress];
				st += " ";
				st += vif[progress++];
				break;

			case LIVE2:

				break;

			case CSIQ:
				st.Empty();
				st += fn;
				st += " ";
				st += strb;
				st += " ";
				st += mos[progress++];
				break;

			case IVC:

				break;

			default:

				break;
			}
			FA.SeekToEnd();
			FA.WriteString(st + "\n");
		} while (!_findnext(h2, &fileinfo) || progress % disimg_num);
	}

	// 预测结束
	_findclose(h2);
	FA.Close();
	pDlg->OnCancel();

	return 0;
}
