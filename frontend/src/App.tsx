import React, {useState} from 'react';
import {InboxOutlined} from '@ant-design/icons';
import {Button, Col, DatePicker, Layout, message, Row, Space, Typography, Upload, type UploadFile} from 'antd';
import axios from 'axios';

const {Dragger} = Upload;
const {Title} = Typography;
const {RangePicker} = DatePicker;
const {Content} = Layout;

const App: React.FC = () => {
    const [fileList, setFileList] = useState<UploadFile[]>([]); // 存储上传的文件列表
    const [processedResults, setProcessedResults] = useState<string[]>([]);
    const [selectedTimeRange, setSelectedTimeRange] = useState<[string, string] | null>(null);
    const [timeCheckResult, setTimeCheckResult] = useState<string | null>(null);
    const [uploading, setUploading] = useState(false);
    const [extractedDocs, setExtractedDocs] = useState([]);
    const uploadProps = {
        name: 'file',
        multiple: true,
        beforeUpload: (file: UploadFile) => {
            // 阻止自动上传
            setFileList((prev) => [...prev, file]);
            return false;
        },
        onRemove: (file: UploadFile) => {
            // 移除文件
            setFileList((prev) => prev.filter((item) => item.uid !== file.uid));
        },
    };

    // 点击上传按钮时触发的逻辑
    const handleUpload = async () => {
        if (fileList.length === 0) {
            message.warning('请先选择文件！');
            return;
        }

        setUploading(true); // 开始上传时设置按钮为加载状态

        try {
            const formData = new FormData();

            // 遍历文件列表并添加到 FormData
            fileList.forEach((file) => {
                // console.log('上传的文件:', file); // 调试信息
                formData.append('files', file); // 直接使用 File 对象
            });

            // 发送 POST 请求到后端服务
            const response = await axios.post('http://127.0.0.1:8000/api/v1/process_files', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data', // 确保使用正确的 Content-Type
                },
            });
            console.log(response);


            if (response.data && response.data.results) {
                setProcessedResults(response.data.results);
                setExtractedDocs(response.data.data);
                message.success('文件上传成功并处理完成！');
            } else {
                message.error('后端未返回处理结果，请检查后端代码！');
            }
        } catch (error: any) {
            console.error('文件上传失败:', error);

            if (error.response && error.response.status === 422) {
                message.error('文件格式错误或请求数据不符合后端要求！');
            } else {
                message.error('文件上传失败，请检查网络或后端服务！');
            }
        } finally {
            setUploading(false); // 上传完成后恢复按钮状态
        }
    };


    // 时间选择框的回调
    const handleTimeChange = (dates: any, dateStrings: [string, string]) => {
        setSelectedTimeRange(dateStrings);
    };

    const checkValidity = async () => {
        if (!selectedTimeRange) {
            message.warning('请先选择时间范围！');
            return;
        }
        if (!extractedDocs || extractedDocs.length === 0) {
            message.warning('请先提取文档信息！');
            return;
        }

        try {
            // 调用后端接口，传递 start_date、end_date 和 docs 参数
            const response = await axios.post('http://127.0.0.1:8000/api/v1/check_validity', {
                start_date: selectedTimeRange[0], // 起始日期
                end_date: selectedTimeRange[1],  // 结束日期
                docs: extractedDocs, // 前一个调用的结果（文档结构化信息）
            });
            // 更新状态，将后端返回的 valid_docs 直接展示在时间选择后的结果区域
            if (response.data && response.data.valid_docs) {
                const validDocs = response.data.valid_docs.map((doc: any, index: number) => (
                    <p key={index}>
                        文件名: {doc["文件名"]}, 类型: {doc["类型"]}, 专利号: {doc["专利号"]},
                        申请日期：{doc["申请日期"]}, 授权日期: {doc["授权日期"]}
                    </p>
                ));
                setTimeCheckResult(validDocs); // 将格式化后的结果更新到状态
                message.success(`时间范围检查完成，总计有效文档: ${response.data.total_valid}`);
            } else {
                message.error('后端未返回有效文档结果，请检查后端代码！');
            }
        } catch (error) {
            console.error('检查有效性失败:', error);
            message.error('检查有效性失败，请重试！');
        }
    };


    return (
        <Layout style={{height: '98vh', width: '100vw', padding: '16px'}}>
            <Content style={{padding: '16px'}}>
                <Row gutter={[16, 16]} style={{height: '100%'}}>
                    {/* 左侧上传区域 */}
                    <Col span={8} style={{
                        border: '1px solid #d9d9d9',
                        padding: '16px',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between'
                    }}>
                        <Title level={5}>上传框</Title>
                        <Dragger {...uploadProps}>
                            <p className="ant-upload-drag-icon">
                                <InboxOutlined/>
                            </p>
                            <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
                            <p className="ant-upload-hint">支持单个或批量上传。</p>
                        </Dragger>
                        <Button
                            type="primary"
                            onClick={handleUpload}
                            loading={uploading} // 按钮的加载状态
                            style={{marginTop: '16px'}}
                        >
                            {uploading ? '处理中...' : '上传'}
                        </Button>
                    </Col>

                    {/* 右侧结果与时间选择区域 */}
                    <Col span={16} style={{padding: '16px'}}>
                        <Row style={{height: '40%', marginBottom: '16px'}}>
                            <Col span={24}>
                                <Title level={5}>上传结果</Title>
                                <div style={{
                                    border: '1px solid #d9d9d9',
                                    height: '25vh',
                                    overflow: 'auto',
                                    padding: '8px'
                                }}>
                                    {processedResults.length > 0 ? (
                                        processedResults.map((result, index) => <p key={index}>{result}</p>)
                                    ) : (
                                        <p>暂无处理结果</p>
                                    )}
                                </div>
                            </Col>
                        </Row>
                        <Row style={{
                            display: 'flex',
                            alignItems: 'center'
                        }}>
                            <Col span={20}>
                                <Space direction="horizontal" size={12}>
                                    选择有效时间:<RangePicker onChange={handleTimeChange}/>
                                </Space>
                            </Col>
                            <Col span={4}>
                                <Button type="primary" onClick={checkValidity} style={{justifyContent: "center"}}>
                                    检查有效文档
                                </Button>
                            </Col>
                        </Row>
                        <Row style={{height: '35%', paddingTop: '16px'}}>
                            <Col span={24}>
                                <Title level={5}>时间选择后的结果</Title>
                                <div style={{
                                    border: '1px solid #d9d9d9',
                                    height: '100%',
                                    overflow: 'auto',
                                    padding: '8px'
                                }}>
                                    {timeCheckResult ? timeCheckResult : <p>暂无时间检查结果</p>}
                                </div>
                            </Col>
                        </Row>

                    </Col>
                </Row>
            </Content>
        </Layout>
    );
};

export default App;
