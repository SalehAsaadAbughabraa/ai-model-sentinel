import { AWSProvider } from '../../src/cloud/providers/aws/aws-provider';
import { CloudProviderConfig, DeploymentConfig, CloudStorageConfig } from '../../src/cloud/types';

// مثال التهيئة الأساسية
async function basicAWSUsage() {
  console.log('🚀 Starting AWS Provider Example...\n');

  // 1. التهيئة
  const config: CloudProviderConfig = {
    provider: 'aws',
    credentials: {
      accessKeyId: process.env.AWS_ACCESS_KEY_ID || 'your-access-key',
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || 'your-secret-key',
      region: process.env.AWS_REGION || 'us-east-1'
    }
  };

  const awsProvider = new AWSProvider(config);

  try {
    // 2. التهيئة والمصادقة
    console.log('📡 Initializing AWS Provider...');
    await awsProvider.initialize();
    
    console.log('🔐 Authenticating with AWS...');
    const isAuthenticated = await awsProvider.authenticate();
    console.log(`✅ Authentication: ${isAuthenticated ? 'Success' : 'Failed'}\n`);

    if (!isAuthenticated) {
      throw new Error('AWS authentication failed');
    }

    // 3. رفع نموذج إلى S3
    console.log('📤 Uploading model to S3...');
    const storageConfig: CloudStorageConfig = {
      bucketName: 'your-model-bucket',
      filePath: 'models/model-v1.h5',
      isPublic: false
    };

    const s3Url = await awsProvider.uploadModel(storageConfig);
    console.log(`✅ Model uploaded to: ${s3Url}\n`);

    // 4. نشر النموذج
    console.log('🚀 Deploying model to AWS Lambda...');
    const deploymentConfig: DeploymentConfig = {
      modelPath: s3Url,
      target: 'lambda',
      runtime: 'python',
      memorySize: 512,
      timeout: 30
    };

    const deploymentId = await awsProvider.deployModel(deploymentConfig);
    console.log(`✅ Model deployed with ID: ${deploymentId}\n`);

    // 5. المراقبة
    console.log('📊 Setting up monitoring...');
    await awsProvider.setupMonitoring({
      metrics: ['invocations', 'errors', 'duration'],
      alertThresholds: {
        errors: 5,
        duration: 1000
      },
      notificationEmails: ['admin@your-company.com']
    });
    console.log('✅ Monitoring setup completed\n');

    // 6. الحصول على القياسات
    console.log('📈 Getting metrics...');
    const metrics = await awsProvider.getMetrics(
      deploymentId,
      new Date(Date.now() - 24 * 60 * 60 * 1000), // آخر 24 ساعة
      new Date()
    );
    console.log('📊 Metrics:', JSON.stringify(metrics, null, 2));
    console.log('✅ Metrics retrieved successfully\n');

    // 7. الحالة الصحية
    console.log('❤️ Health check...');
    const isHealthy = await awsProvider.healthCheck();
    console.log(`✅ Health status: ${isHealthy ? 'Healthy' : 'Unhealthy'}\n`);

    console.log('🎉 All AWS operations completed successfully!');

  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

// تشغيل المثال
basicAWSUsage();