// Load modules
const express = require('express')
const path = require('path')
const bodyParser = require('body-parser')
const pythonShell = require('python-shell')
const fileUpload = require('express-fileupload')

// Init App
const app = express()

// Load View Engine
app.set('views', path.join(__dirname, 'views'))
app.set('view engine', 'pug')

// File
app.use(fileUpload())

// Body Parser Middleware
// parse application/x-www-form-urlencoded
app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser.json())

// Set Public Folder
app.use(express.static(path.join(__dirname, 'public')))

// Home Route
app.get('/', function(req, res) {
	res.render('index')
})

// Recommend Route
app.post('/classify', function(req, res) {
	if(!req.files)
		return res.status(400).send('No files were uploaded.')

	// console.log(req.files.sampleFile.name)

	// The name of the input field (i.e. "sampleFile") is used to retrieve the uploaded file
	let sampleFile = req.files.sampleFile
	let fileName = req.files.sampleFile.name

	// Use the mv() method to place the file somewhere on your server
	let path = 'audio-files/' + fileName
	// console.log(path)
	sampleFile.mv(path, function(err) {
		if(err)
			return res.status(500).send(err)
	})

	// Call twitter-scraper python script
	var options = {
		mode: 'text',
		pythonPath: 'C:/Users/alant/Anaconda3/python.exe',
		scriptPath: './',
		args:
		[
			fileName,	// music file path
		]
	}

	pythonShell.run('../classifiers/linear-svm-classifier/linear-svm-demo.py', options, function(err, results) {
		if(err) {
			throw err
		} else {
			let jsonResults = JSON.parse(results)
			console.log(jsonResults)

			res.render('index', {
				audio_file_name: jsonResults.filename,
				image: jsonResults.image,
				label: jsonResults.label,
				features: jsonResults.features,
				values: jsonResults.values,
				music_results: results,
				confidence: jsonResults.confidence
			})
		}
	})
})

// Start server on port 3000
app.listen(8080, function() {
	console.log('Server started on port 8080...')
})
