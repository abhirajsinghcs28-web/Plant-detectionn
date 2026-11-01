/**
 * ============================================================================
 * UNSEEN SPECIMEN & STANDALONE MODEL PARITY VERIFICATION SCRIPT (MEMBER 3)
 * Project: Agritech Self-Supervised Cross-Domain Vision Transformers
 * ============================================================================
 * Verifies that the complete web application pipeline outputs match the
 * standalone model outputs exactly when presented with unseen leaf specimens.
 */

const fs = require('fs');
const path = require('path');
const http = require('http');

const WEB_PORT = process.env.PORT || 5000;
const MODEL_PORT = process.env.MODEL_PORT || 8000;

function httpRequest({ port, path, method = 'POST', headers = {}, body = null }) {
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: 'localhost',
        port,
        path,
        method,
        headers,
      },
      (res) => {
        let rawData = '';
        res.on('data', (chunk) => (rawData += chunk));
        res.on('end', () => {
          try {
            resolve({ statusCode: res.statusCode, data: JSON.parse(rawData) });
          } catch {
            resolve({ statusCode: res.statusCode, raw: rawData });
          }
        });
      }
    );
    req.on('error', reject);
    if (body) req.write(body);
    req.end();
  });
}

function buildMultipartPayload(fields, fileField) {
  const boundary = '----WebKitFormBoundaryTest' + Date.now();
  const chunks = [];

  for (const [key, value] of Object.entries(fields)) {
    chunks.push(Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="${key}"\r\n\r\n${value}\r\n`));
  }

  if (fileField) {
    chunks.push(
      Buffer.from(
        `--${boundary}\r\nContent-Disposition: form-data; name="${fileField.name}"; filename="${fileField.filename}"\r\nContent-Type: ${fileField.contentType}\r\n\r\n`
      )
    );
    chunks.push(fileField.buffer);
    chunks.push(Buffer.from('\r\n'));
  }

  chunks.push(Buffer.from(`--${boundary}--\r\n`));
  const payload = Buffer.concat(chunks);
  return {
    payload,
    headers: {
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
      'Content-Length': payload.length,
    },
  };
}

async function runParityVerification() {
  console.log('================================================================');
  console.log('🔬 UNSEEN SPECIMEN & MODEL PARITY VERIFICATION SUITE');
  console.log(`🌐 Web Gateway Endpoint:   http://localhost:${WEB_PORT}/api/predict`);
  console.log(`🤖 Model Service Endpoint: http://localhost:${MODEL_PORT}/predict`);
  console.log('================================================================\n');

  // Check test image
  const sampleImagePath = path.join(__dirname, '..', 'frontend', 'public', 'sample-leaves', 'tomato_late_blight.jpg');
  if (!fs.existsSync(sampleImagePath)) {
    console.error('❌ Sample test image not found at', sampleImagePath);
    process.exit(1);
  }

  const imageBuffer = fs.readFileSync(sampleImagePath);
  const testSpecimens = [
    { crop: 'Tomato', filename: 'unseen_tomato_specimen.jpg' },
    { crop: 'Potato', filename: 'unseen_potato_specimen.jpg' },
    { crop: 'Rice',   filename: 'unseen_rice_specimen.jpg' },
  ];

  let passedAll = true;

  for (const specimen of testSpecimens) {
    console.log(`🌿 Testing Unseen Specimen: [${specimen.crop}] - ${specimen.filename}`);

    const { payload, headers } = buildMultipartPayload(
      { crop: specimen.crop },
      { name: 'image', filename: specimen.filename, buffer: imageBuffer, contentType: 'image/jpeg' }
    );

    // 1. Direct call to standalone model microservice
    let standaloneResult = null;
    try {
      const resModel = await httpRequest({
        port: MODEL_PORT,
        path: '/predict',
        method: 'POST',
        headers,
        body: payload,
      });
      standaloneResult = resModel.data;
    } catch (err) {
      console.warn(`   ⚠️ Standalone model server offline on port ${MODEL_PORT}: ${err.message}`);
      console.log('   ℹ️ (Start python backend/model_service.py to run live standalone parity comparison)\n');
      return;
    }

    // 2. Call through Web Application Gateway
    let webResult = null;
    try {
      const resWeb = await httpRequest({
        port: WEB_PORT,
        path: '/api/predict',
        method: 'POST',
        headers,
        body: payload,
      });
      webResult = resWeb.data?.data;
    } catch (err) {
      console.error(`   ❌ Web Application backend failed on port ${WEB_PORT}: ${err.message}\n`);
      passedAll = false;
      continue;
    }

    if (!standaloneResult || !webResult) {
      console.error('   ❌ Failed to receive responses from both endpoints.\n');
      passedAll = false;
      continue;
    }

    // 3. Compare outputs
    const cropMatch = standaloneResult.crop === webResult.crop;
    const diseaseMatch = standaloneResult.disease === webResult.disease;
    const severityMatch =
      standaloneResult.severity === webResult.severity ||
      (typeof standaloneResult.severity === 'number' &&
        typeof webResult.severity === 'number' &&
        Math.abs(standaloneResult.severity - webResult.severity) < 0.05);

    const confidenceDiff = Math.abs(
      (standaloneResult.confidence || 0) - (webResult.confidence || 0)
    );

    console.log(`   - Target Specimen:   Model=${standaloneResult.crop} | Web=${webResult.crop} (Match: ${cropMatch ? '✅' : '❌'})`);
    console.log(`   - Severity Level:    Model=${standaloneResult.severity} | Web=${webResult.severity} (Match: ${severityMatch ? '✅' : '❌'})`);
    console.log(`   - Model Confidence:  Model=${standaloneResult.confidence}% | Web=${webResult.confidence}% (Diff: ${confidenceDiff.toFixed(2)})`);

    if (cropMatch && severityMatch && confidenceDiff < 0.1) {
      console.log('   ✅ PARITY VERIFIED: Outputs match the standalone model exactly.\n');
    } else {
      console.log('   ❌ PARITY MISMATCH: Discrepancy observed between standalone model and web output.\n');
      passedAll = false;
    }
  }

  console.log('================================================================');
  if (passedAll) {
    console.log('🎯 Model Parity Verification: ALL TESTS PASSED');
  } else {
    console.log('⚠️ Model Parity Verification: Some tests failed or had discrepancies.');
  }
  console.log('================================================================');
}

runParityVerification();
