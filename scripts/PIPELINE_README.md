# Automated Video Processing Pipeline

Automatically processes videos uploaded to the Ready folder through the complete multi-modal search pipeline.

## What It Does

When a video is added to `s3://tl-brice-media/WBD_project/Videos/Ready/`, the pipeline:

1. **Generates embeddings** via AWS Bedrock Marengo 3.0
   - Visual embeddings (512d per 6-second segment)
   - Audio embeddings (512d per 6-second segment)
   - Transcription embeddings (512d per 6-second segment)

2. **Stores in MongoDB Atlas** (single-index backend)
   - Stores each modality separately with `modality_type` filter
   - Enables fast vector search with MongoDB Atlas Vector Search

3. **Stores in S3 Vectors** (multi-index backend)
   - Separate indexes for visual, audio, and transcription
   - Enables modality-specific vector search optimization

4. **Moves video to proxy folder**
   - Moves from `Ready/` to `proxy/` for CloudFront delivery
   - Updates all stored s3_uri references to proxy path
   - App automatically serves videos from proxy location

## Prerequisites

```bash
# 1. AWS SSO login
aws sso login --profile TlFullDevelopmentAccess-026090552520

# 2. Set MongoDB URI
export MONGODB_URI='mongodb+srv://...'

# 3. Install dependencies
pip install boto3 pymongo
```

## Usage

### Process a Single Video

```bash
python scripts/process_video_pipeline.py s3://tl-brice-media/WBD_project/Videos/Ready/my_video.mp4
```

### Watch Mode (Continuous Processing)

```bash
# Check every 60 seconds (default)
python scripts/process_video_pipeline.py --watch

# Check every 30 seconds
python scripts/process_video_pipeline.py --watch --interval 30
```

### Manual Upload Process

1. **Upload video to Ready folder:**
   ```bash
   aws s3 cp my_video.mp4 s3://tl-brice-media/WBD_project/Videos/Ready/
   ```

2. **Run pipeline** (or wait if watch mode is running):
   ```bash
   python scripts/process_video_pipeline.py s3://tl-brice-media/WBD_project/Videos/Ready/my_video.mp4
   ```

3. **Video is automatically processed and moved to proxy/**

4. **Search in the app** at https://nyfwaxmgni.us-east-1.awsapprunner.com

## Output Example

```
============================================================
Processing: s3://tl-brice-media/WBD_project/Videos/Ready/friends_s01e01.mp4
============================================================

[1/4] Generating embeddings via Bedrock Marengo 3.0...
Status: Completed (attempt 45)
✓ Generated embeddings for 150 segments

[2/4] Storing embeddings in MongoDB Atlas...
✓ Stored 450 embeddings in MongoDB

[3/4] Storing embeddings in S3 Vectors...
✓ Stored in S3 Vectors:
  - Visual: 150 segments
  - Audio: 150 segments
  - Transcription: 150 segments

[4/4] Moving video to proxy folder...
✓ Moved to: s3://tl-brice-media/WBD_project/Videos/proxy/friends_s01e01.mp4

============================================================
✓ Processing complete for friends_s01e01
============================================================
```

## Automated Processing (Lambda)

For fully automated processing, deploy the pipeline as a Lambda function triggered by S3 events:

1. **Create Lambda function** from `process_video_pipeline.py`
2. **Add S3 trigger** for `WBD_project/Videos/Ready/` prefix
3. **Set environment variables:**
   - `MONGODB_URI`: Your MongoDB connection string
   - `AWS_REGION`: us-east-1

4. **Configure IAM permissions:**
   - S3: Read from Ready/, Write/Delete to Ready/, Write to proxy/
   - Bedrock: InvokeModel permissions for Marengo 3.0
   - S3 Vectors: PutVectors permissions
   - CloudWatch Logs: For Lambda logging

## Troubleshooting

### AWS SSO Token Expired
```bash
aws sso login --profile TlFullDevelopmentAccess-026090552520
```

### MongoDB Connection Failed
```bash
# Verify MongoDB URI is set
echo $MONGODB_URI

# Test connection
mongosh "$MONGODB_URI"
```

### Bedrock Processing Timeout
- Marengo processing can take 5-10 minutes for long videos
- Lambda timeout should be set to 15 minutes maximum
- For very long videos (>30 min), use the script directly instead of Lambda

### S3 Vectors Storage Failed
```bash
# Verify S3 Vectors indexes exist
aws s3vectors list-indexes \
  --vector-bucket-name brice-video-search-multimodal \
  --region us-east-1
```

## File Structure

```
s3://tl-brice-media/
└── WBD_project/
    └── Videos/
        ├── Ready/          ← Upload videos here
        │   └── [processing happens here]
        └── proxy/          ← Videos moved here after processing
            └── [app serves from here]
```

## Cost Estimates

Per video (15 minutes, 6-second segments = 150 segments):

- **Bedrock Marengo 3.0**: ~$0.50-1.00 per video
- **MongoDB Atlas**: ~$0.01 per video (storage + queries)
- **S3 Vectors**: ~$0.02 per video (storage + queries)
- **S3 Storage**: ~$0.001 per video per month
- **CloudFront**: Pay per GB transferred

**Total**: ~$0.50-1.00 per video for initial processing, ~$0.03/month for storage
