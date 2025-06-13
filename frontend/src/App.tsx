import React, { useState } from 'react';
import { InboxOutlined } from '@ant-design/icons';
import { Upload, message, Layout, Row, Col, Typography, Button, DatePicker, Space } from 'antd';
import type { UploadProps } from 'antd';
import axios from 'axios';

const { Dragger } = Upload;
const { Title } = Typography;
const { RangePicker } = DatePicker;
const { Header, Content } = Layout;

const App: React.FC = () => {
  const [processedResults, setProcessedResults] = useState<string[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState<[string, string] | null>(null);
  const [timeCheckResult, setTimeCheckResult] = useState<string | null>(null);

  // 上传组件的配置
  const uploadProps: UploadProps = {
    name: 'file',
    multiple: true,
    action: 'https://660d2bd96ddfa2943b33731c.mockapi.io/api/upload', // 示例接口，替换为后端接口
    onChange(info) {
      const { status } = info.file;
      if (status !== 'uploading') {
        console.log(info.file, info.fileList);
      }
      if (status === 'done') {
        message.success(`${info.file.name} file uploaded successfully.`);
        processFiles(info.fileList);
      } else if (status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
    onDrop(e) {
      console.log('Dropped files', e.dataTransfer.files);
    },
  };

  // 调用后端接口 /api/v1/process_files
  const processFiles = async (files: any) => {
    try {
      const formData = new FormData();
      files.forEach((file: any) => formData.append('files', file.originFileObj));

      const response = await axios.post('/api/v1/process_files', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setProcessedResults(response.data.results);
      message.success('Files processed successfully!');
    } catch (error) {
      console.error('Error processing files:', error);
      message.error('Failed to process files.');
    }
  };

  // 时间选择框的回调
  const handleTimeChange = (dates: any, dateStrings: [string, string]) => {
    setSelectedTimeRange(dateStrings);
  };

  // 调用后端接口 /api/v1/time_check
  const checkTime = async () => {
    if (!selectedTimeRange) {
      message.warning('Please select a time range first.');
      return;
    }

    try {
      const response = await axios.post('/api/v1/time_check', {
        start_time: selectedTimeRange[0],
        end_time: selectedTimeRange[1],
      });

      setTimeCheckResult(response.data.message);
      message.success(`Time check successful: ${response.data.message}`);
    } catch (error) {
      console.error('Error checking time:', error);
      message.error('Failed to check time.');
    }
  };

  return (
    <Layout style={{ height: '98vh', width:'100vw',padding: '16px' }}>
      <Content style={{ padding: '16px' }}>
        <Row gutter={[16, 16]} style={{ height: '100%' }}>
          {/* 左侧上传区域 */}
          <Col span={8} style={{ border: '1px solid #d9d9d9', padding: '16px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <Title level={5}>上传框</Title>
            <Dragger {...uploadProps}>
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
              <p className="ant-upload-hint">支持单个或批量上传。禁止上传公司数据或其他敏感文件。</p>
            </Dragger>
            <Button type="primary" style={{ marginTop: '16px' }}>
              上传按钮
            </Button>
          </Col>

          {/* 右侧结果与时间选择区域 */}
          <Col span={16} style={{ border: '1px solid #d9d9d9', padding: '16px' }}>
            <Row style={{ height: '40%', borderBottom: '1px solid #d9d9d9', paddingBottom: '16px' }}>
              <Col span={24}>
                <Title level={5}>上传后的结果</Title>
                <div style={{ border: '1px solid #d9d9d9', height: '100%', overflow: 'auto', padding: '8px' }}>
                  {processedResults.length > 0 ? (
                    processedResults.map((result, index) => <p key={index}>{result}</p>)
                  ) : (
                    <p>暂无处理结果</p>
                  )}
                </div>
              </Col>
            </Row>
            <Row style={{ height: '20%', borderBottom: '1px solid #d9d9d9', paddingBottom: '16px', display: 'flex', alignItems: 'center' }}>
              <Col span={20}>
                <Title level={5}>时间选择器</Title>
                <Space direction="vertical" size={12}>
                  <RangePicker showTime onChange={handleTimeChange} />
                </Space>
              </Col>
              <Col span={4}>
                <Button type="primary" onClick={checkTime}>
                  选择按钮
                </Button>
              </Col>
            </Row>
            <Row style={{ height: '35%', paddingTop: '16px' }}>
              <Col span={24}>
                <Title level={5}>时间选择后的结果</Title>
                <div style={{ border: '1px solid #d9d9d9', height: '100%', overflow: 'auto', padding: '8px' }}>
                  {timeCheckResult ? <p>{timeCheckResult}</p> : <p>暂无时间检查结果</p>}
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
