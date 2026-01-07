import './App.css'
import Canvas from './Canvas.jsx'
import { useRef } from 'react'
import { Button } from 'react-bootstrap';

function App() {
  // useRef is used instead of useState in order to prevent 
  // component being reloaded
  const canvasRef = useRef(null);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  return (
    <>
      <div>
        <p>Draw some numbers in the canvas then click classify!</p>
      </div>
      <Canvas canvasRef={canvasRef} />
      <div className="d-flex gap-2">
        <Button variant="success" size="lg">Classify Numbers</Button>
        <Button onClick={clearCanvas} variant="primary" size="lg">Clear Canvas</Button>
      </div>
    </>
  )
}

export default App
