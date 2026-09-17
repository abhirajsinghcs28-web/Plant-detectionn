const express = require('express');
const cors = require('cors');
const path = require('path');
const mongoose = require('mongoose');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

const predictRoutes = require('./routes/predictRoutes');
const errorHandler = require('./middleware/errorHandler');

const app = express();
const PORT = process.env.PORT || 5000;
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/plant_disease_db';

// Enable CORS for frontend clients
app.use(
  cors({
    origin: [
      'http://localhost:5173',
      'http://127.0.0.1:5173',
      'http://localhost:5174',
      'http://127.0.0.1:5174',
      'http://localhost:3000',
    ],
    credentials: true,
  })
);

// Standard parsers
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve uploaded leaf images statically
const uploadDir = path.join(__dirname, 'uploads');
app.use('/uploads', express.static(uploadDir));

const aiService = require('./services/aiService');

// System Health Check Endpoint
app.get('/api/health', async (req, res) => {
  const mongoStatus =
    mongoose.connection.readyState === 1
      ? 'connected'
      : mongoose.connection.readyState === 2
      ? 'connecting'
      : 'disconnected';

  const aiStatus = await aiService.checkAIHealth();

  res.status(200).json({
    status: 'online',
    project: 'Agritech Crop Leaf Disease Severity Estimation',
    timestamp: new Date().toISOString(),
    database: mongoStatus,
    aiService: aiStatus,
  });
});

// API Routes
app.use('/api', predictRoutes);

// Centralized error handling
app.use(errorHandler);

// Connect to MongoDB
mongoose
  .connect(MONGODB_URI, {
    serverSelectionTimeoutMS: 5000,
  })
  .then(() => {
    console.log(`[MongoDB] Connected successfully to: ${MONGODB_URI}`);
  })
  .catch((err) => {
    console.warn(`[MongoDB Warning] Could not connect to MongoDB: ${err.message}`);
    console.warn('[MongoDB Warning] The server will still run and provide predictions, but history persistence may be unavailable.');
  });

// Start listening with resilient retry if port is in TIME_WAIT
function startServer(portToUse, retryCount = 0) {
  const server = app.listen(portToUse, () => {
    console.log('====================================================');
    console.log(`🌱 Agritech Leaf Disease Backend Server running!`);
    console.log(`📡 URL: http://localhost:${portToUse}`);
    console.log(`🏥 Health check: http://localhost:${portToUse}/api/health`);
    console.log(`🚀 Predict API: POST http://localhost:${portToUse}/api/predict`);
    console.log(`📁 Static uploads: http://localhost:${portToUse}/uploads`);
    console.log('====================================================');
  });

  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      if (retryCount < 5) {
        console.log(`⏳ Port ${portToUse} is in TIME_WAIT. Automatically retrying in 2 seconds (attempt ${retryCount + 1}/5)...`);
        setTimeout(() => {
          startServer(portToUse, retryCount + 1);
        }, 2000);
      } else {
        console.error(`\n❌ [Server Error] Port ${portToUse} is occupied by another application.`);
        console.error(`💡 Hint: Close any other terminal or app using port ${portToUse}.\n`);
      }
    } else {
      console.error('❌ [Server Error]', err);
    }
  });
}

startServer(PORT);


