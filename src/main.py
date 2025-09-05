from pathlib import Path
import yaml

def main():
    project_root = Path(__file__).parent.parent.resolve()
    
    yaml_path = project_root / 'data' / 'Weeds' / 'data.yaml'
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    
    print("Conteúdo do data.yaml:")
    print(data_yaml)

if __name__ == "__main__":
    main()
