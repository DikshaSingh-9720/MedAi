import gradio as gr
import io
import asyncio
from PIL import Image

from app.main import predict


def hf_predict(image, modality):
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        class DummyFile:
            def __init__(self, file):
                self.file = file
                self.content_type = "image/png"

            async def read(self):
                return self.file.read()

        file = DummyFile(buf)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            predict(file=file, modality=modality)
        )

        return result

    except Exception as e:
        return {"error": str(e)}


interface = gr.Interface(
    fn=hf_predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(["xray", "ct", "ultrasound", "mri"], value="xray")
    ],
    outputs="json",
    title="MedAI Diagnostic System",
    description="Upload medical image and get AI prediction"
)

interface.launch()