const fs = require('fs');
const http = require('http');
const path = require('path');

const samplePath = path.join(__dirname, '..', 'frontend', 'public', 'sample-leaves', 'tomato_late_blight.jpg');
const filePath = fs.existsSync(samplePath) ? samplePath : path.join(__dirname, 'uploads', 'sample.jpg');
const boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW';

const fileContent = fs.readFileSync(filePath);
const fileName = 'tomato_sample.jpg';

const payloadHeader = 
  `--${boundary}\r\n` +
  `Content-Disposition: form-data; name="crop"\r\n\r\n` +
  `Tomato\r\n` +
  `--${boundary}\r\n` +
  `Content-Disposition: form-data; name="image"; filename="${fileName}"\r\n` +
  `Content-Type: image/jpeg\r\n\r\n`;

const payloadFooter = `\r\n--${boundary}--\r\n`;

const payload = Buffer.concat([
  Buffer.from(payloadHeader, 'utf-8'),
  fileContent,
  Buffer.from(payloadFooter, 'utf-8')
]);

const options = {
  hostname: 'localhost',
  port: 5000,
  path: '/api/predict',
  method: 'POST',
  headers: {
    'Content-Type': `multipart/form-data; boundary=${boundary}`,
    'Content-Length': payload.length
  }
};

const req = http.request(options, (res) => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    console.log('Status Code:', res.statusCode);
    console.log('Response:', JSON.stringify(JSON.parse(data), null, 2));
  });
});

req.on('error', (err) => {
  console.error('Request Error:', err.message);
});

req.write(payload);
req.end();
