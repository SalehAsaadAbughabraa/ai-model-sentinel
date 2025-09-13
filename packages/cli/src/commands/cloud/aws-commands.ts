import { AWSProvider } from '@ai-model-sentinel/core';
import { CloudProviderConfig, DeploymentConfig, CloudStorageConfig } from '@ai-model-sentinel/core';

export class AWSCommands {
  private provider: AWSProvider;

  constructor(config: CloudProviderConfig) {
    this.provider = new AWSProvider(config);
  }

  async setup(accessKeyId: string, secretAccessKey: string, region: string) {
    console.log('🔧 Setting up AWS credentials...');
    
    try {
      await this.provider.initialize();
      const authenticated = await this.provider.authenticate();
      
      if (authenticated) {
        console.log('✅ AWS credentials configured successfully');
        console.log(`📍 Region: ${region}`);
        console.log('🔐 Authentication: Valid');
      } else {
        console.log('❌ AWS authentication failed');
      }
    } catch (error) {
      console.error('❌ Setup failed:', error.message);
    }
  }

  async deployModel(modelPath: string, bucketName: string, runtime: string) {
    console.log('🚀 Deploying model to AWS...');
    
    try {
      await this.provider.initialize();

      // 1. Upload to S3
      console.log('📤 Uploading model to S3...');
      const storageConfig: CloudStorageConfig = {
        bucketName,
        filePath: `models/${Date.now()}/model`,
        isPublic: false
      };

      const s3Url = await this.provider.uploadModel(storageConfig);
      console.log(`✅ Model uploaded to: ${s3Url}`);

      // 2. Deploy to Lambda
      console.log('⚡ Deploying to AWS Lambda...');
      const deploymentConfig: DeploymentConfig = {
        modelPath: s3Url,
        target: 'lambda',
        runtime: runtime as any,
        memorySize: 512,
        timeout: 30
      };

      const deploymentId = await this.provider.deployModel(deploymentConfig);
      console.log(`🎉 Model deployed successfully!`);
      console.log(`📋 Deployment ID: ${deploymentId}`);
      console.log(`🌐 S3 Location: ${s3Url}`);

      return deploymentId;

    } catch (error) {
      console.error('❌ Deployment failed:', error.message);
      throw error;
    }
  }

  async setupMonitoring(deploymentId: string) {
    console.log('📊 Setting up monitoring...');
    
    try {
      await this.provider.initialize();

      await this.provider.setupMonitoring({
        metrics: ['invocations', 'errors', 'duration'],
        alertThresholds: {
          errors: 5,
          duration: 1000
        },
        notificationEmails: [process.env.ADMIN_EMAIL || 'admin@your-company.com']
      });

      console.log('✅ Monitoring setup completed');
      console.log(`📈 Monitoring deployment: ${deploymentId}`);

    } catch (error) {
      console.error('❌ Monitoring setup failed:', error.message);
    }
  }

  async listDeployments() {
    console.log('📋 Listing deployments...');
    
    try {
      await this.provider.initialize();
      const deployments = await this.provider.listDeployments();
      
      if (deployments.length === 0) {
        console.log('📭 No deployments found');
      } else {
        console.log('🚀 Active deployments:');
        deployments.forEach((deployment, index) => {
          console.log(`${index + 1}. ${deployment}`);
        });
      }

      return deployments;

    } catch (error) {
      console.error('❌ Failed to list deployments:', error.message);
    }
  }

  async checkStatus(deploymentId: string) {
    console.log(`🔍 Checking status of ${deploymentId}...`);
    
    try {
      await this.provider.initialize();
      const status = await this.provider.getDeploymentStatus(deploymentId);
      
      console.log(`📊 Deployment Status: ${status}`);
      return status;

    } catch (error) {
      console.error('❌ Failed to check status:', error.message);
    }
  }

  async healthCheck() {
    console.log('❤️ Performing AWS health check...');
    
    try {
      await this.provider.initialize();
      const isHealthy = await this.provider.healthCheck();
      
      console.log(`✅ Health Status: ${isHealthy ? 'Healthy' : 'Unhealthy'}`);
      return isHealthy;

    } catch (error) {
      console.error('❌ Health check failed:', error.message);
      return false;
    }
  }
}