package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
)

var direccion_nodo string

const (
	puerto_registro = 8000
	puerto_notifica = 8001
	puerto_proceso  = 8002
)

var bitacora []string

func main() {
	//Identificación = IP
	direccion_nodo = descubreIP() //"192.168.101.12"
	fmt.Printf("IP = %s\n", direccion_nodo)
	//Server
	go registrarServidor()
	go registrarProceso()
	//Client
	//A que nodo se requiere unir???
	bufferIn := bufio.NewReader(os.Stdin)
	fmt.Print("Ingrese el Ip del nodo a unir: ")
	strDirNodoRem, _ := bufferIn.ReadString('\n')
	strDirNodoRem = strings.TrimSpace(strDirNodoRem)
	if strDirNodoRem != "" {
		registrarCliente(strDirNodoRem)
	}
	//servidor
	escucharNotificaciones()
}

func registrarServidor() {
	//formatear el host para la conexion  ip:port
	hostname := fmt.Sprintf("%s:%d", direccion_nodo, puerto_registro)
	ln, _ := net.Listen("tcp", hostname)
	defer ln.Close()
	for {
		//aceptar las conexiones
		conn, _ := ln.Accept()
		//manejar la conexion concurrentemente
		go manejadorRegistro(conn)
	}
}
func manejadorRegistro(conn net.Conn) {
	defer conn.Close()
	//recuperar el mensaje de registro (IP de nodo)
	//Leer el Ip del nodo q solicitó unirse
	bufferIn := bufio.NewReader(conn)
	msgIP, _ := bufferIn.ReadString('\n')
	msgIP = strings.TrimSpace(msgIP)
	//serializar la bitacora de direcciones
	bytesBitacora, _ := json.Marshal(bitacora)
	fmt.Fprintf(conn, "%s\n", string(bytesBitacora)) //escribiendo al cliente
	notificarTodos(msgIP)                            //Pull del mensaje hacia todos los nodos registrados en la bitacora
	bitacora = append(bitacora, msgIP)               //Agregar la ip que llega como mensaje, a la bitacora
	fmt.Println(bitacora)                            //imprimir la bitacora
}
func notificarTodos(msgIP string) {
	//hace un pull del mensaje
	for _, direccion := range bitacora {
		notificar(direccion, msgIP)
	}
}
func notificar(direccion, msgIP string) {
	remotehost := fmt.Sprintf("%s:%d", direccion, puerto_notifica)
	conn, _ := net.Dial("tcp", remotehost)
	defer conn.Close()
	fmt.Fprintf(conn, "%s\n", msgIP) //enviar el msgIp por la conexión del cliente
}
func registrarCliente(strDirNodoRem string) {
	remotehost := fmt.Sprintf("%s:%d", strDirNodoRem, puerto_registro)
	conn, _ := net.Dial("tcp", remotehost)
	defer conn.Close()
	//el cliente envía su dirección IP
	fmt.Fprintf(conn, "%s\n", direccion_nodo)
	//recibe la bitacora del servidor
	bufferIn := bufio.NewReader(conn)
	msgBitacora, _ := bufferIn.ReadString('\n')
	//deserializar
	var arrBitacora []string
	json.Unmarshal([]byte(msgBitacora), &arrBitacora)
	bitacora = append(arrBitacora, strDirNodoRem) //actualiza la bitacora de direcciones ip del cluster
	fmt.Println(bitacora)
}
func escucharNotificaciones() {
	hostname := fmt.Sprintf("%s:%d", direccion_nodo, puerto_notifica)
	ln, _ := net.Listen("tcp", hostname)
	defer ln.Close()
	for {
		conn, _ := ln.Accept()
		go manejadorNotificaciones(conn)
	}
}
func manejadorNotificaciones(conn net.Conn) {
	defer conn.Close()
	//recuperar el mensaje IP
	bufferIn := bufio.NewReader(conn)
	msgIP, _ := bufferIn.ReadString('\n')
	msgIP = strings.TrimSpace(msgIP)
	//agregarlo a la bitacora
	bitacora = append(bitacora, msgIP)
	fmt.Println(bitacora)
}
func descubreIP() string {
	listaInterfaces, _ := net.Interfaces()
	for _, interf := range listaInterfaces {
		//fmt.Println(interf.Name)
		direcciones, _ := interf.Addrs()
		for _, direccion := range direcciones {
			//fmt.Println(direccion)
			switch d := direccion.(type) {
			case *net.IPNet:
				//fmt.Println(d.IP)
				if strings.HasPrefix(d.IP.String(), "192") {
					//fmt.Println(d.IP)
					return d.IP.String()
				}
			}

		}
	}
	return ""
}

func registrarProceso() {
	hostname := fmt.Sprintf("%s:%d", direccion_nodo, puerto_proceso)
	ln, _ := net.Listen("tcp", hostname) //IP:Port
	defer ln.Close()
	for {
		conn, _ := ln.Accept()
		go manejarProceso(conn)
	}
}

// Recibe Numero y predice
func manejarProceso(conn net.Conn) {
	defer conn.Close()
	bufferIn := bufio.NewReader(conn)
	strNum, _ := bufferIn.ReadString('\n')
	strNum = strings.TrimSpace(strNum)
	num, _ := strconv.Atoi(strNum)
	fmt.Printf("Numero recibido: %d\n", num)
	if num == 0 {
		fmt.Println("Boommm!!!")
	} else {
		goProceso()
	}
}
