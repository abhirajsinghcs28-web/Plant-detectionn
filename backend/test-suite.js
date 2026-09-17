/**
 * ============================================================================
 * AUTOMATED SYSTEM INTEGRATION TEST SUITE (MEMBER 3)
 * Project: Agritech Crop Leaf Disease Severity Estimation
 * ============================================================================
 * Tests:
 * 1. GET /api/health (System status, MongoDB state, AI service mode)
 * 2. POST /api/predict (Valid submission, mock provenance check, zero fake mask check)
 * 3. POST /api/predict (Missing image validation error check)
 * 4. POST /api/predict (Invalid crop species whitelist error check)
 * 5. GET /api/predictions (MongoDB historical record retrieval)
 * 6. DELETE /api/predictions/:id (MongoDB record cleanup)
 */

const fs = require('fs');
const path = require('path');
const http = require('http');

const PORT = process.env.PORT || 5000;
const HOST = 'localhost';

function httpRequest({ method, path, headers = {}, body = null }) {
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: HOST,
        port: PORT,
        path,
        method,
        headers,
      },
      (res) => {
        let rawData = '';
        res.on('data', (chunk) => (rawData += chunk));
        res.on('end', () => {
          try {
            const parsed = rawData ? JSON.parse(rawData) : null;
            resolve({ statusCode: res.statusCode, headers: res.headers, data: parsed, rawText: rawData });
          } catch {
            resolve({ statusCode: res.statusCode, headers: res.headers, data: null, rawText: rawData });
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
  let chunks = [];

  for (const [key, value] of Object.entries(fields)) {
    chunks.push(
      Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="${key}"\r\n\r\n${value}\r\n`)
    );
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

async function runTests() {
  console.log('===========================================================');
  console.log('🌱 Starting Member 3 Integration Test Suite...');
  console.log(`📡 Target Backend: http://${HOST}:${PORT}`);
  console.log('===========================================================\n');

  let passed = 0;
  let failed = 0;
  let createdRecordId = null;

  // TEST 1: Health check
  try {
    const res = await httpRequest({ method: 'GET', path: '/api/health' });
    if (
      res.statusCode === 200 &&
      res.data?.status === 'online' &&
      res.data?.database === 'connected' &&
      res.data?.aiService?.mode === 'mock'
    ) {
      console.log('✅ TEST 1 PASSED: GET /api/health reported online, MongoDB connected, and AI mock mode.');
      passed++;
    } else {
      console.error('❌ TEST 1 FAILED: Unexpected health check output:', res.data);
      failed++;
    }
  } catch (err) {
    console.error('❌ TEST 1 FAILED with exception:', err.message);
    failed++;
  }

  // Create a minimal 1x1 JPG buffer for test submissions
  // 1x1 valid JPEG byte sequence:
  const dummyJpgBuffer = Buffer.from([
    0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43,
    0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
    0x09, 0x08, 0x0a, 0x0c, 0x14, 0x0d, 0x0c, 0x0b, 0x0b, 0x0c, 0x19, 0x12,
    0x13, 0x0f, 0x14, 0x1d, 0x1a, 0x1f, 0x1e, 0x1d, 0x1a, 0x1c, 0x1c, 0x20,
    0x24, 0x2e, 0x27, 0x20, 0x22, 0x2c, 0x23, 0x1c, 0x1c, 0x28, 0x37, 0x29,
    0x2c, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1f, 0x27, 0x39, 0x3d, 0x38, 0x32,
    0x3c, 0x2e, 0x33, 0x34, 0x32, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x01,
    0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xff, 0xc4, 0x00, 0x1f, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0a, 0x0b, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3f,
    0x00, 0xbf, 0x80, 0xff, 0xd9
  ]);

  // TEST 2: POST /api/predict (Valid image & crop)
  try {
    const { payload, headers } = buildMultipartPayload(
      { crop: 'Tomato' },
      { name: 'image', filename: 'valid_test_leaf.jpg', contentType: 'image/jpeg', buffer: dummyJpgBuffer }
    );

    const res = await httpRequest({ method: 'POST', path: '/api/predict', headers, body: payload });
    if (
      res.statusCode === 200 &&
      res.data?.success === true &&
      res.data?.data?.isMock === true &&
      res.data?.data?.segmentationMask === null &&
      typeof res.data?.data?.severity === 'number'
    ) {
      createdRecordId = res.data.data.id;
      console.log('✅ TEST 2 PASSED: POST /api/predict correctly generated mock result with isMock:true & segmentationMask:null.');
      passed++;
    } else {
      console.error('❌ TEST 2 FAILED: Prediction response does not conform to contract:', res.data);
      failed++;
    }
  } catch (err) {
    console.error('❌ TEST 2 FAILED with exception:', err.message);
    failed++;
  }

  // TEST 3: POST /api/predict (Missing image error)
  try {
    const { payload, headers } = buildMultipartPayload({ crop: 'Tomato' }, null);
    const res = await httpRequest({ method: 'POST', path: '/api/predict', headers, body: payload });
    if (res.statusCode === 400 && res.data?.success === false && res.data?.error?.includes('No leaf image')) {
      console.log('✅ TEST 3 PASSED: Missing leaf image correctly rejected with HTTP 400.');
      passed++;
    } else {
      console.error('❌ TEST 3 FAILED: Server did not reject missing image:', res.data);
      failed++;
    }
  } catch (err) {
    console.error('❌ TEST 3 FAILED with exception:', err.message);
    failed++;
  }

  // TEST 4: POST /api/predict (Invalid crop species whitelist error)
  try {
    const { payload, headers } = buildMultipartPayload(
      { crop: 'Banana' }, // Banana is not in the whitelist
      { name: 'image', filename: 'banana_test.jpg', contentType: 'image/jpeg', buffer: dummyJpgBuffer }
    );

    const res = await httpRequest({ method: 'POST', path: '/api/predict', headers, body: payload });
    if (res.statusCode === 400 && res.data?.success === false && res.data?.error?.includes('Invalid crop species')) {
      console.log('✅ TEST 4 PASSED: Non-whitelisted crop correctly rejected with HTTP 400.');
      passed++;
    } else {
      console.error('❌ TEST 4 FAILED: Server accepted non-whitelisted crop species:', res.data);
      failed++;
    }
  } catch (err) {
    console.error('❌ TEST 4 FAILED with exception:', err.message);
    failed++;
  }

  // TEST 5: GET /api/predictions (History retrieval)
  try {
    const res = await httpRequest({ method: 'GET', path: '/api/predictions' });
    if (res.statusCode === 200 && res.data?.success === true && Array.isArray(res.data?.data)) {
      console.log(`✅ TEST 5 PASSED: GET /api/predictions retrieved ${res.data.count} records from MongoDB.`);
      passed++;
    } else {
      console.error('❌ TEST 5 FAILED: Could not retrieve prediction records:', res.data);
      failed++;
    }
  } catch (err) {
    console.error('❌ TEST 5 FAILED with exception:', err.message);
    failed++;
  }

  // TEST 6: DELETE /api/predictions/:id (Cleanup)
  if (createdRecordId) {
    try {
      const res = await httpRequest({ method: 'DELETE', path: `/api/predictions/${createdRecordId}` });
      if (res.statusCode === 200 && res.data?.success === true) {
        console.log(`✅ TEST 6 PASSED: DELETE /api/predictions/${createdRecordId} cleaned up test record.`);
        passed++;
      } else {
        console.error('❌ TEST 6 FAILED: Could not delete record:', res.data);
        failed++;
      }
    } catch (err) {
      console.error('❌ TEST 6 FAILED with exception:', err.message);
      failed++;
    }
  }

  console.log('\n===========================================================');
  console.log(`🎯 Test Summary: ${passed} Passed, ${failed} Failed`);
  console.log('===========================================================');

  if (failed > 0) {
    process.exit(1);
  }
}

runTests();
