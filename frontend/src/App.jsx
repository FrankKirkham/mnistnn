import './App.css'
import Canvas from './Canvas.jsx'
import { useRef, useState } from 'react'
import { Button } from 'react-bootstrap';

function App() {
  // useRef is used instead of useState in order to prevent 
  // component being reloaded
  const canvasRef = useRef(null);
  const [classifyDisabled, setClassifyDisabled] = useState(false);

  const clearCanvas = () => {
    // Allow us to classify again
    setClassifyDisabled(false)

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  const classifyNumbersOnCanvas = () => {
    // Disable this button until cleared
    setClassifyDisabled(true)

    const canvas = canvasRef.current;

    // Convert the canvas to a useable image (BLOB)
    canvas.toBlob(async (blob) => {
      // Format the image to be added to a JSON body
      const formData = new FormData();
      formData.append("image", blob, "canvas.png");

      // Send the image to backend to be classified
      const response = await fetch("http://localhost:5000/classify", {
        method: "POST",
        body: formData,
      });

      // Get the data from the response
      const data = await response.json();
      // Update the canvas with the info
      drawInfoOnCanvas(data); 
    }, "image/png");
  }

  function drawInfoOnCanvas(data) {
    data.forEach((obj) => {
      const threshold = 0.20
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      // Display the box around the image
      const coords = obj["location"];
      // Expand the bounding box by 1 and check it doesn't go out of bounds
      // const x = Math.min(ctx.width - 1, Math.max(0, (coords[1] + 1)));
      // const y = Math.min(ctx.height - 1, Math.max(0, (coords[0] + 1)));
      // const w = Math.min(ctx.width - 1 - x, (coords[3] - coords[1]) + 1);
      // const h = Math.min(ctx.height - 1 - y, (coords[2] - coords[0]) + 1);
      const x = Math.max(0, (coords[1] - 5));
      const y = Math.max(0, (coords[0] - 5));
      const w = (coords[3] - x) + 5;
      const h = (coords[2] - y) + 5;
      
      ctx.strokeStyle = 'green';
      ctx.strokeRect(x, y, w, h);
      // Get and print the results above the threshold
      const percents = obj["results"].filter(result => result[1] >= threshold);
      for(let i = 0; i < percents.length; i++) {
        console.log(i)
        ctx.fillStyle = 'green';
        ctx.fillText(`${percents[i][0]} : ${(100 * percents[i][1]).toFixed(1)}%`, x, y + h + (10 * (i + 1)))
      }

      // NOTE: TEMP!!
      // Change back for drawining on canvas
      ctx.strokeStyle = 'black';

      // print values above threshold
    });
  }

  return (
    <>
      <div>
        <p>Draw some numbers in the canvas then click classify!</p>
      </div>
      <Canvas canvasRef={canvasRef} />
      <div className="d-flex gap-2">
        <Button onClick={classifyNumbersOnCanvas} 
                disabled={classifyDisabled} variant="success" size="lg">Classify Numbers</Button>
        <Button onClick={clearCanvas} variant="primary" size="lg">Clear Canvas</Button>
      </div>
    </>
  )
}

export default App
