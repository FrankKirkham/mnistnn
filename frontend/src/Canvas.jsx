import { useRef, useEffect } from 'react'

function Canvas({canvasRef}) {
    // useRef is used instead of useState in order to prevent 
    // component being reloaded
    const isDrawing = useRef(false);

    // Runs once component loaded
    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        ctx.lineWidth = 2;
        ctx.lineCap = "round";
        ctx.strokeStyle = "black";
    }, [])

    const startDrawing = (e) => {
        isDrawing.current = true;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        ctx.beginPath();
        ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    }

    const draw = (e) => {
        if (!isDrawing.current) {
            return;
        }

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
        ctx.stroke();
    }

    const stopDrawing = () => {
        isDrawing.current = false;
    }

    return (
        <canvas
            ref={canvasRef}
            width={800}
            height={300}
            style={{width: 800, height: 300 }} // "touchAction: 'none'"
            className="border border-3 border-primary rounded shadow"
            onPointerDown={startDrawing}
            onPointerMove={draw}
            onPointerUp={stopDrawing}
            onPointerLeave={stopDrawing}
        />
    )
}

export default Canvas