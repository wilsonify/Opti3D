# Project Title

Optimization of 3D Printable Objects

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```sh
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
python3 setup.py install
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

```
python3 -m pip install test-requirements.txt
pytest
```

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc



# Opti3D Web Application

A modern web interface for optimizing STL files for 3D printing.

## Features

- **Drag & Drop Upload**: Simply drag your STL file onto the upload area
- **File Analysis**: View detailed information about your STL file including triangles, vertices, and dimensions
- **Multiple Optimization Levels**:
  - Light: Minimal optimization, preserves maximum quality
  - Medium: Balanced optimization and quality
  - Aggressive: Maximum file reduction, may affect quality
- **Real-time Processing**: See optimization progress with live updates
- **Download Optimized Files**: Get your optimized STL file ready for 3D printing

## Quick Start

1. **Install Dependencies**:
   ```bash
   cd src
   python -m pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Open Browser**:
   Navigate to `http://localhost:5000`

## Usage

1. Upload your STL file by dragging and dropping or clicking the upload area
2. View the file analysis showing triangles, vertices, and dimensions
3. Choose an optimization level (Light, Medium, or Aggressive)
4. Click "Optimize File" to process your STL
5. Download your optimized file for 3D printing

## API Endpoints

- `GET /` - Main web interface
- `POST /api/upload` - Upload STL file for analysis
- `POST /api/optimize` - Optimize uploaded file
- `GET /api/download/<filename>` - Download optimized file
- `POST /api/cleanup` - Clean up temporary files

## File Requirements

- **Format**: STL files only
- **Size**: Maximum 100MB
- **Binary/ASCII**: Both formats supported

## Optimization Details

### Light Optimization
- Removes degenerate triangles (zero area)
- Preserves original mesh structure
- Minimal file size reduction

### Medium Optimization
- Removes degenerate triangles
- Merges duplicate vertices (tolerance: 0.01mm)
- Significant file size reduction with quality preservation

### Aggressive Optimization
- Removes degenerate triangles
- Merges duplicate vertices (tolerance: 0.1mm)
- Applies Laplacian smoothing
- Maximum file size reduction

## Technical Details

- **Backend**: Flask web framework
- **STL Processing**: numpy-stl library
- **Frontend**: Modern HTML5 with TailwindCSS
- **File Storage**: Temporary files with automatic cleanup
- **Security**: File type validation and size limits

## Development

### Project Structure
```
src/
├── app.py                 # Main Flask application
├── templates/
│   ├── base.html         # Base template
│   └── index.html        # Main interface
├── stldeli/
│   ├── stl_optimizer.py  # STL optimization algorithms
│   └── ...
└── requirements.txt      # Python dependencies
```

### Adding New Optimization Features

1. Add new optimization functions to `stldeli/stl_optimizer.py`
2. Update the API endpoint in `app.py` to support new options
3. Modify the frontend interface to expose new settings

## License

This project is licensed under the MIT License - see the LICENSE file for details.
