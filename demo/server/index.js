const express = require('express');
const multer = require('multer');
const path = require('path');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/results', express.static(path.join(__dirname, 'results')));

// Ensure directories exist
['uploads', 'results'].forEach(dir => {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir);
    }
});

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, `${Date.now()}-${file.originalname}`);
    }
});

const upload = multer({ storage: storage });

app.post('/api/restore', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).send('No file uploaded.');
    }

    const inputPath = req.file.path;
    const outputPath = path.join('results', `restored-${req.file.filename}`);

    console.log(`Starting restoration for: ${inputPath}`);

    // Call Python script
    const pythonPath = process.env.PYTHON_PATH || '../../training/venv/bin/python';
    const scriptPath = process.env.INFERENCE_SCRIPT_PATH || '../../training/scripts/inference.py';

    const pythonProcess = spawn(pythonPath, [
        scriptPath,
        '--input', path.resolve(inputPath),
        '--output', path.resolve(outputPath)
    ]);

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            res.json({
                success: true,
                restoredImage: `/results/restored-${req.file.filename}`,
                originalImage: `/uploads/${req.file.filename}`
            });
        } else {
            res.status(500).json({ success: false, error: 'Restoration failed' });
        }
    });
});

app.listen(port, () => {
    console.log(`Restoration server running at http://localhost:${port}`);
});
