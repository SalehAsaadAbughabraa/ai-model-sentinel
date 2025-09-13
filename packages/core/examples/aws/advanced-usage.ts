import { AWSProvider } from '../../src/cloud/providers/aws/aws-provider';
import { CloudProviderConfig, DeploymentConfig, CloudStorageConfig, MonitoringConfig } from '../../src/cloud/types';

// مثال متقدم مع معالجة الأخطاء
async function advancedAWSUsage() {
  console.log('🚀 Starting Advanced AWS Provider Example...\n');

  const config: CloudProviderConfig = {
    provider: 'aws',
    credentials: {
      accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
      region: process.env.AWS_REGION || 'us-east-1'
    }
  };

  const awsProvider = new AWSProvider(config);

  try {
    // التهيئة
    await awsProvider.initialize();
    
    // المصادقة
    if (!await awsProvider.authenticate()) {
      throw new Error('Authentication failed - check your AWS credentials');
    }

    // 1. رفع نموذج متعدد
    console.log('📤 Uploading multiple model versions...');
    
    const modelVersions = ['v1', 'v2', 'v3'];
    const uploadPromises = modelVersions.map(async (version) => {
      const storageConfig: CloudStorageConfig = {
        bucketName: 'ai-model-bucket',
        filePath: `models/image-classifier/${version}/model.h5`,
        isPublic: false
      };
      
      const url = await awsProvider.uploadModel(storageConfig);
      console.log(`✅ ${version} uploaded: ${url}`);
      return url;
    });

    const modelUrls = await Promise.all(uploadPromises);
    console.log('✅ All models uploaded successfully\n');

    // 2. نشر متعدد
    console.log('🚀 Deploying multiple models...');
    
    const deploymentPromises = modelUrls.map(async (modelUrl, index) => {
      const deploymentConfig: DeploymentConfig = {
        modelPath: modelUrl,
        target: 'lambda',
        runtime: 'python',
        memorySize: 1024,
        timeout: 60
      };

      const deploymentId = await awsProvider.deployModel(deploymentConfig);
      console.log(`✅ Model ${modelVersions[index]} deployed: ${deploymentId}`);
      return deploymentId;
    });

    const deploymentIds = await Promise.all(deploymentPromises);
    console.log('✅ All models deployed successfully\n');

    // 3. إعداد مراقبة متقدمة
    console.log('📊 Setting up advanced monitoring...');
    
    const monitoringConfig: MonitoringConfig = {
      metrics: [
        'invocations',
        'errors', 
        'duration',
        'throttles',
        'concurrentExecutions'
      ],
      alertThresholds: {
        errors: 10,
        duration: 5000,
        throttles: 5,
        concurrentExecutions: 100
      },
      notificationEmails: [
        'ai-team@your-company.com',
        'devops@your-company.com'
      ]
    };

    await awsProvider.setupMonitoring(monitoringConfig);
    console.log('✅ Advanced monitoring setup completed\n');

    // 4. إدارة النشرات
    console.log('📋 Listing deployments...');
    const deployments = await awsProvider.listDeployments();
    console.log('📊 Active deployments:', deployments);
    console.log('✅ Deployment list retrieved\n');

    // 5. حالة النشر
    console.log('🔍 Checking deployment status...');
    for (const deploymentId of deploymentIds) {
      const status = await awsProvider.getDeploymentStatus(deploymentId);
      console.log(`📋 ${deploymentId}: ${status}`);
    }
    console.log('✅ All deployment statuses checked\n');

    console.log('🎉 Advanced AWS operations completed successfully!');

  } catch (error) {
    console.error('❌ Advanced example error:', error.message);
    console.error('Stack:', error.stack);
  }
}

// تشغيل المثال المتقدم
advancedAWSUsage();