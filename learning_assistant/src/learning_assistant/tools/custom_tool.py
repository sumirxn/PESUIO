from crewai_tools import BaseTool
import base64
import requests
import json
import os
from typing import Optional



class MyCustomTool(BaseTool):
    name: str = "FlowchartTool"
    description: str = (
        "It is a flowchart generating tool. It makes a mind map of all the important information to help the user understand visually. This tool also generates diagrams when necessary. "
    )

    def mermaid_to_image(mermaid_code: str, output_path: Optional[str] = None, img_type: str = 'png') -> str:
        """
        Convert Mermaid diagram code to an image using Mermaid's online API.
        
        Args:
            mermaid_code (str): The Mermaid diagram code as a string
            output_path (str, optional): Path where to save the image. If None, generates a default path
            img_type (str): Image format - 'png' or 'svg'. Defaults to 'png'
        
        Returns:
            str: Path to the saved image file
            
        Raises:
            ValueError: If invalid image type specified
            RequestException: If API request fails
        """
        if img_type not in ['png', 'svg']:
            raise ValueError("img_type must be either 'png' or 'svg'")
        
        # Clean up the Mermaid code
        mermaid_code = mermaid_code.strip()
        
        # Encode the Mermaid code to base64
        encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
        
        # Create API URL
        api_url = f"https://mermaid.ink/img/{encoded}"
        
        # Make the request
        response = requests.get(api_url)
        response.raise_for_status()
        
        # Generate default output path if none provided
        if output_path is None:
            output_path = f"mermaid_diagram.{img_type}"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save the image
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return output_path

    # Example usage
    if __name__ == "__main__":
        # Example Mermaid code
        mermaid_code = """
        graph TD
            A[Start] --> B{Is it?}
            B -- Yes --> C[OK]
            C --> D[Rethink]
            D --> B
            B -- No --> E[End]
        """
        
        try:
            # Convert to PNG
            png_path = mermaid_to_image(mermaid_code, "diagram.png")
            print(f"PNG diagram saved to: {png_path}")
            
            # Convert to SVG
            svg_path = mermaid_to_image(mermaid_code, "diagram.svg", img_type='svg')
            print(f"SVG diagram saved to: {svg_path}")
            
        except requests.RequestException as e:
            print(f"Error making API request: {e}")
        except Exception as e:
            print(f"Error: {e}")