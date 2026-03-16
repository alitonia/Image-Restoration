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

app.post('/api/restore', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.status(400).send('No file uploaded.');
    }

    const { method = 'swinir' } = req.body;
    const inputPath = req.file.path;
    const results = {};

    const runInference = (script, outputSuffix, extraArgs = []) => {
        return new Promise((resolve, reject) => {
            const outputPath = path.join('results', `${outputSuffix}-${req.file.filename}`);
            const pythonPath = process.env.PYTHON_PATH || '../../training/venv/bin/python';
            const scriptPath = path.join('../../training/scripts', script);

            console.log(`Running ${script} for: ${inputPath}`);
            const pythonProcess = spawn(pythonPath, [
                scriptPath,
                '--input', path.resolve(inputPath),
                '--output', path.resolve(outputPath),
                ...extraArgs
            ]);

            pythonProcess.on('close', (code) => {
                if (code === 0) resolve(`/results/${outputSuffix}-${req.file.filename}`);
                else reject(new Error(`${script} failed with code ${code}`));
            });
        });
    };

    try {
        if (method === 'swinir' || method === 'both') {
            results.swinir = await runInference('inference_swinir.py', 'swinir');
        }
        if (method === 'sd' || method === 'both') {
            results.sd = await runInference('inference_sd.py', 'sd');
        }

        res.json({
            success: true,
            originalImage: `/uploads/${req.file.filename}`,
            results
        });
    } catch (error) {
        console.error('Restoration Error:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

app.listen(port, () => {
    console.log(`Restoration server running at http://localhost:${port}`);
});
