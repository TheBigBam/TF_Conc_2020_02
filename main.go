package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"image"
	"image/png"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

var red Red

func uploadFile(w http.ResponseWriter, r *http.Request) {
	fmt.Println("File Upload Endpoint Hit")

	r.ParseMultipartForm(10 << 20)

	file, handler, err := r.FormFile("myFile")
	if err != nil {
		fmt.Println("Error Retrieving the File")
		fmt.Println(err)
		return
	}
	defer file.Close()
	fmt.Printf("Uploaded File: %+v\n", handler.Filename)
	fmt.Printf("File Size: %+v\n", handler.Size)
	fmt.Printf("MIME Header: %+v\n", handler.Header)

	tempFile, err := ioutil.TempFile("temp-images", "upload-*.png")
	if err != nil {
		fmt.Println(err)
	}
	defer tempFile.Close()

	fileBytes, err := ioutil.ReadAll(file)
	if err != nil {
		fmt.Println(err)
	}

	tempFile.Write(fileBytes)

	fmt.Fprintf(w, "Successfully Uploaded File\n")
	var temp = nn(tempFile.Name())
	fmt.Fprintf(w, "Numero Predecido:")
	fmt.Fprintf(w, temp)
}

func setupRoutes() {
	http.HandleFunc("/upload", uploadFile)
	http.ListenAndServe(":8080", nil)
}

func goProceso() {
	fmt.Println("Hello World")

	red := crearRed(784, 200, 10, 0.1)

	fmt.Println("Entrenar")
	mnistEntrenarConcurrent(&red)
	save(red)

	fmt.Println("Predicir")
	load(&red)
	mnistPredict(&red)

	setupRoutes()
}

func main() {
	fmt.Println("Hello World")

	red := crearRed(784, 200, 10, 0.1)

	fmt.Println("Entrenar")
	mnistEntrenarConcurrent(&red)
	save(red)

	fmt.Println("Predicir")
	load(&red)
	mnistPredict(&red)

	setupRoutes()
}

func nn(file string) string {
	red := crearRed(784, 200, 10, 0.1)
	if file != "" {
		// load the neural network from file
		load(&red)
		// predict which number it is
		fmt.Println("prediction:", predictFromImage(red, file))
		return strconv.Itoa(predictFromImage(red, file))
	}
	return ""
}

func mnistEntrenarConcurrent(red *Red) {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()

	// # de Epocas
	const N = 5

	var wg sync.WaitGroup
	wg.Add(N)

	for epocas := 0; epocas < N; epocas++ {
		epocas := epocas
		go func() {
			fmt.Println("Epocas: ", epocas)
			mnistEntrenar(red)
			fmt.Println("Done: ", epocas)
			wg.Done()
		}()
	}
	wg.Wait()

	elapsed := time.Since(t1)
	fmt.Printf("\n Tiempo para Entrenar: %s \n", elapsed)
}

func mnistEntrenar(red *Red) {
	testArch, _ := os.Open("NN/mnist_dataset/mnist_train.csv")
	r := csv.NewReader(bufio.NewReader(testArch))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}

		entrada := make([]float64, red.entrada)
		for i := range entrada {
			x, _ := strconv.ParseFloat(record[i], 64)
			entrada[i] = (x / 255.0 * 0.999) + 0.001
		}
		objetivo := make([]float64, 10)

		for i := range objetivo {
			objetivo[i] = 0.001
		}
		x, _ := strconv.Atoi(record[0])
		objetivo[x] = 0.999

		red.Entrenar(entrada, objetivo)
	}
	testArch.Close()
}

func mnistPredict(red *Red) {
	t1 := time.Now()
	testArch, _ := os.Open("NN/mnist_dataset/mnist_test.csv")
	defer testArch.Close()

	puntaje := 0
	total := 0
	r := csv.NewReader(bufio.NewReader(testArch))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		entrada := make([]float64, red.entrada)
		for i := range entrada {
			if i == 0 {
				entrada[i] = 1.0
			}
			x, _ := strconv.ParseFloat(record[i], 64)
			entrada[i] = (x / 255.0 * 0.999) + 0.001
		}
		salidas := red.Predict(entrada)
		mejor := 0
		max := 0.0
		for i := 0; i < red.salidas; i++ {
			if salidas.At(i, 0) > max {
				mejor = i
				max = salidas.At(i, 0)
			}
		}
		objetivo, _ := strconv.Atoi(record[0])
		if mejor == objetivo {
			puntaje++
		}
		total++
	}

	elapsed := time.Since(t1)
	var correcto = (float64(puntaje) / float64(total)) * 100

	fmt.Printf("Tiempo para predecir: %s\n", elapsed)
	fmt.Println("puntaje: ", puntaje)
	fmt.Println("total: ", total)
	fmt.Println("% correcto: ", correcto)
}

// Una red de 3 capas ocultas (entrada, oculto y salida)
type Red struct {
	entrada          int
	ocultos          int
	salidas          int
	pesosOcultos     *mat.Dense
	pesosSalida      *mat.Dense
	ratioAprendizaje float64
}

// crea una red con pesos aleatorios.
func crearRed(entrada, oculto, salida int, rate float64) (red Red) {
	red = Red{
		entrada:          entrada,
		ocultos:          oculto,
		salidas:          salida,
		ratioAprendizaje: rate,
	}
	red.pesosOcultos = mat.NewDense(red.ocultos, red.entrada, randomArray(red.entrada*red.ocultos, float64(red.entrada)))
	red.pesosSalida = mat.NewDense(red.salidas, red.ocultos, randomArray(red.ocultos*red.salidas, float64(red.ocultos)))
	return
}

// Entrenar la red neuronal
func (red *Red) Entrenar(entradaData []float64, objetivoData []float64) {
	// feedforward
	entrada := mat.NewDense(len(entradaData), 1, entradaData)
	ocultoentrada := dot(red.pesosOcultos, entrada)
	ocultosalidas := apply(sigmoid, ocultoentrada)
	finalentrada := dot(red.pesosSalida, ocultosalidas)
	finalsalidas := apply(sigmoid, finalentrada)

	// encontrar errores
	objetivo := mat.NewDense(len(objetivoData), 1, objetivoData)
	salidaErrors := subtract(objetivo, finalsalidas)
	ocultoErrors := dot(red.pesosSalida.T(), salidaErrors)

	// backpropagate
	red.pesosSalida = add(red.pesosSalida,
		scale(red.ratioAprendizaje,
			dot(multiply(salidaErrors, sigmoidPrime(finalsalidas)),
				ocultosalidas.T()))).(*mat.Dense)

	red.pesosOcultos = add(red.pesosOcultos,
		scale(red.ratioAprendizaje,
			dot(multiply(ocultoErrors, sigmoidPrime(ocultosalidas)),
				entrada.T()))).(*mat.Dense)
}

// Predice usando la red neuronal el numero
func (red Red) Predict(entradaData []float64) mat.Matrix {
	entrada := mat.NewDense(len(entradaData), 1, entradaData)
	ocultoentrada := dot(red.pesosOcultos, entrada)
	ocultosalidas := apply(sigmoid, ocultoentrada)
	finalentrada := dot(red.pesosSalida, ocultosalidas)
	finalsalidas := apply(sigmoid, finalentrada)
	return finalsalidas
}

func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, subtract(ones, m)) // m * (1 - m)
}

func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
}

func addBiasNodeTo(m mat.Matrix, b float64) mat.Matrix {
	r, _ := m.Dims()
	a := mat.NewDense(r+1, 1, nil)

	a.Set(0, 0, b)
	for i := 0; i < r; i++ {
		a.Set(i+1, 0, m.At(i, 0))
	}
	return a
}

func save(red Red) {
	h, err := os.Create("NN/data/hweights.model")
	defer h.Close()
	if err == nil {
		red.pesosOcultos.MarshalBinaryTo(h)
	}
	o, err := os.Create("NN/data/oweights.model")
	defer o.Close()
	if err == nil {
		red.pesosSalida.MarshalBinaryTo(o)
	}
}

// load a neural red from file
func load(red *Red) {
	h, err := os.Open("NN/data/hweights.model")
	defer h.Close()
	if err == nil {
		red.pesosOcultos.Reset()
		red.pesosOcultos.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("NN/data/oweights.model")
	defer o.Close()
	if err == nil {
		red.pesosSalida.Reset()
		red.pesosSalida.UnmarshalBinaryFrom(o)
	}
	return
}

// predict a number from an image
// image should be 28 x 28 PNG file
func predictFromImage(net Red, path string) int {
	input := dataFromImage(path)
	output := net.Predict(input)
	best := 0
	highest := 0.0
	for i := 0; i < net.salidas; i++ {
		if output.At(i, 0) > highest {
			best = i
			highest = output.At(i, 0)
		}
	}
	return best
}

func dataFromImage(filePath string) (pixels []float64) {
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	img, err := png.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}

	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var rgba = img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}

	pixels = make([]float64, len(gray.Pix))

	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.999) + 0.001
	}
	return
}
