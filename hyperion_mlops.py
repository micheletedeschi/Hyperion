#!/usr/bin/env python3
"""
🚀 HYPERION3 - SISTEMA MLOPS AVANZADO
Sistema completo de MLOps para gestión de modelos, experimentos y deployment
Versión 3.0 - Arquitectura MLOps Profesional
"""

import os
import json
import pickle
import joblib
import hashlib
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd

# Importación segura de psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.layout import Layout
    from rich.tree import Tree
    from rich import print as rprint
    from utils.safe_progress import Progress
    RICH_AVAILABLE = True
except ImportError:
    Console = None
    RICH_AVAILABLE = False

def make_json_serializable(obj):
    """
    Convierte objetos no serializables a representaciones serializables para JSON.
    """
    # Manejar valores None, NaN e Inf
    if obj is None:
        return None
    elif isinstance(obj, (int, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif hasattr(obj, '__dict__') and hasattr(obj, '__class__'):
        # Objetos complejos como modelos de scikit-learn
        return {
            'type': obj.__class__.__name__,
            'module': obj.__class__.__module__,
            'repr': str(obj)
        }
    elif isinstance(obj, (np.integer, np.floating)):
        val = obj.item()
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

def clean_metrics_for_json(metrics: Dict) -> Dict:
    """
    Limpia un diccionario de métricas removiendo objetos no serializables
    y convirtiendo otros a formatos serializables.
    """
    cleaned = {}
    exclude_keys = {'model_object', 'model', 'estimator', 'trained_model'}
    
    for key, value in metrics.items():
        if key in exclude_keys:
            continue
        
        try:
            # Intentar serializar directamente
            json.dumps(value)
            cleaned[key] = value
        except TypeError:
            # Si no se puede serializar, usar la función helper
            cleaned[key] = make_json_serializable(value)
    
    return cleaned

class MLOpsManager:
    """Gestor completo de MLOps para Hyperion3"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Inicializar sistema MLOps"""
        self.base_dir = Path(base_dir) if base_dir else Path(".")
        self.console = Console() if RICH_AVAILABLE else None
        
        # Directorios MLOps
        self.mlops_dir = self.base_dir / "mlops"
        self.experiments_dir = self.mlops_dir / "experiments"
        self.models_dir = self.mlops_dir / "models"
        self.artifacts_dir = self.mlops_dir / "artifacts"
        self.reports_dir = self.mlops_dir / "reports"
        self.metrics_dir = self.mlops_dir / "metrics"
        self.logs_dir = self.mlops_dir / "logs"
        
        # Crear estructura de directorios
        for dir_path in [self.mlops_dir, self.experiments_dir, self.models_dir, 
                        self.artifacts_dir, self.reports_dir, self.metrics_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Base de datos de experimentos
        self.db_path = self.mlops_dir / "experiments.db"
        self._init_database()
        
        # Estado actual del sistema
        self.current_experiment = None
        self.session_id = self._generate_session_id()
        
    def _init_database(self):
        """Inicializar base de datos de experimentos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de experimentos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_seconds REAL,
                r2_score REAL,
                mse REAL,
                mae REAL,
                rmse REAL,
                parameters TEXT,
                metrics TEXT,
                artifacts_path TEXT,
                model_path TEXT,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de métricas por época
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS epoch_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                epoch INTEGER,
                loss REAL,
                val_loss REAL,
                r2_score REAL,
                learning_rate REAL,
                timestamp TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        # Tabla de recursos del sistema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                cpu_percent REAL,
                memory_percent REAL,
                memory_used_gb REAL,
                gpu_memory_used_gb REAL,
                timestamp TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _generate_session_id(self):
        """Generar ID único de sesión"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def start_experiment(self, name: str, model_type: str, category: str, parameters: Optional[Dict] = None) -> str:
        """Iniciar nuevo experimento"""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        # Crear directorio del experimento
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Guardar en base de datos
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experiments (id, name, model_type, category, status, start_time, parameters, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_id, name, model_type, category, 'running', 
            datetime.now(), json.dumps(clean_metrics_for_json(parameters or {})), self.session_id
        ))
        
        conn.commit()
        conn.close()
        
        self.current_experiment = {
            'id': experiment_id,
            'name': name,
            'model_type': model_type,
            'category': category,
            'start_time': datetime.now(),
            'dir': exp_dir
        }
        
        # Log inicio
        self._log_experiment_event("Experimento iniciado", {"experiment_id": experiment_id})
        
        if self.console:
            self.console.print(f"[green]🚀 Experimento iniciado: {experiment_id}[/green]")
        
        return experiment_id
    
    def log_epoch_metrics(self, experiment_id: str, epoch: int, metrics: Dict):
        """Registrar métricas de época"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO epoch_metrics (experiment_id, epoch, loss, val_loss, r2_score, learning_rate, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_id, epoch, 
            metrics.get('loss'), metrics.get('val_loss'), 
            metrics.get('r2_score'), metrics.get('learning_rate'),
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
    
    def log_system_metrics(self, experiment_id: str):
        """Registrar métricas del sistema"""
        try:
            if not PSUTIL_AVAILABLE:
                return
                
            # CPU y memoria
            cpu_percent = psutil.cpu_percent()  # type: ignore
            memory = psutil.virtual_memory()   # type: ignore
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            
            # GPU (si está disponible)
            gpu_memory_used_gb = 0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
            except:
                pass
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_metrics (experiment_id, cpu_percent, memory_percent, memory_used_gb, gpu_memory_used_gb, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id, cpu_percent, memory_percent, 
                memory_used_gb, gpu_memory_used_gb, datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self._log_experiment_event("Error registrando métricas del sistema", {"error": str(e)})
    
    def finish_experiment(self, experiment_id: str, final_metrics: Dict, model_object=None, artifacts: Optional[Dict] = None):
        """Finalizar experimento y aplicar MLOps completo"""
        if not self.current_experiment or self.current_experiment['id'] != experiment_id:
            if self.console:
                self.console.print(f"[red]❌ Experimento {experiment_id} no encontrado o no activo[/red]")
            return
        
        end_time = datetime.now()
        duration = (end_time - self.current_experiment['start_time']).total_seconds()
        
        # Actualizar base de datos
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE experiments 
            SET status = ?, end_time = ?, duration_seconds = ?, 
                r2_score = ?, mse = ?, mae = ?, rmse = ?, metrics = ?
            WHERE id = ?
        ''', (
            'completed', end_time, duration,
            final_metrics.get('r2_score'), final_metrics.get('mse'),
            final_metrics.get('mae'), final_metrics.get('rmse'),
            json.dumps(clean_metrics_for_json(final_metrics)), experiment_id
        ))
        
        conn.commit()
        conn.close()
        
        # Guardar modelo si se proporciona
        model_path = None
        if model_object is not None:
            model_path = self._save_model(experiment_id, model_object)
        
        # Guardar artefactos
        artifacts_path = None
        if artifacts:
            artifacts_path = self._save_artifacts(experiment_id, artifacts)
        
        # Generar reporte completo
        report_path = self._generate_experiment_report(experiment_id, final_metrics, model_path, artifacts_path)
        
        # Comparar con experimentos anteriores
        comparison = self._compare_with_previous_experiments(experiment_id)
        
        # Mostrar resumen final
        self._show_experiment_summary(experiment_id, final_metrics, comparison, report_path)
        
        # Limpiar experimento actual
        self.current_experiment = None
        
        return {
            'experiment_id': experiment_id,
            'model_path': model_path,
            'artifacts_path': artifacts_path,
            'report_path': report_path,
            'comparison': comparison
        }
    
    def _save_model(self, experiment_id: str, model_object) -> str:
        """Guardar modelo entrenado"""
        model_dir = self.models_dir / experiment_id
        model_dir.mkdir(exist_ok=True)
        
        # Determinar formato de guardado
        model_path = model_dir / "model"
        
        try:
            # Intentar con joblib primero (sklearn)
            joblib.dump(model_object, str(model_path) + ".joblib")
            model_path = str(model_path) + ".joblib"
        except:
            try:
                # Intentar con pickle
                with open(str(model_path) + ".pkl", 'wb') as f:
                    pickle.dump(model_object, f)
                model_path = str(model_path) + ".pkl"
            except:
                try:
                    # Intentar con torch (PyTorch)
                    import torch
                    torch.save(model_object, str(model_path) + ".pth")
                    model_path = str(model_path) + ".pth"
                except:
                    # Guardar estado como JSON
                    with open(str(model_path) + ".json", 'w') as f:
                        json.dump({"model_type": str(type(model_object)), "saved": False}, f)
                    model_path = str(model_path) + ".json"
        
        # Actualizar base de datos con ruta del modelo
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE experiments SET model_path = ? WHERE id = ?', (model_path, experiment_id))
        conn.commit()
        conn.close()
        
        return model_path
    
    def _save_artifacts(self, experiment_id: str, artifacts: Dict) -> str:
        """Guardar artefactos del experimento"""
        artifacts_dir = self.artifacts_dir / experiment_id
        artifacts_dir.mkdir(exist_ok=True)
        
        artifacts_manifest = []
        
        for name, content in artifacts.items():
            artifact_path = artifacts_dir / f"{name}.json"
            
            try:
                if isinstance(content, (dict, list)):
                    with open(artifact_path, 'w') as f:
                        json.dump(content, f, indent=2, default=str)
                elif isinstance(content, pd.DataFrame):
                    content.to_csv(str(artifact_path).replace('.json', '.csv'), index=False)
                    artifact_path = str(artifact_path).replace('.json', '.csv')
                elif isinstance(content, np.ndarray):
                    np.save(str(artifact_path).replace('.json', '.npy'), content)
                    artifact_path = str(artifact_path).replace('.json', '.npy')
                else:
                    with open(artifact_path, 'w') as f:
                        json.dump({"content": str(content)}, f)
                
                artifacts_manifest.append({
                    "name": name,
                    "path": str(artifact_path),
                    "type": str(type(content)),
                    "size": os.path.getsize(artifact_path) if os.path.exists(artifact_path) else 0
                })
                
            except Exception as e:
                artifacts_manifest.append({
                    "name": name,
                    "error": str(e),
                    "type": str(type(content))
                })
        
        # Guardar manifiesto
        manifest_path = artifacts_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(artifacts_manifest, f, indent=2)
        
        # Actualizar base de datos
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE experiments SET artifacts_path = ? WHERE id = ?', (str(artifacts_dir), experiment_id))
        conn.commit()
        conn.close()
        
        return str(artifacts_dir)
    
    def _generate_experiment_report(self, experiment_id: str, metrics: Dict, model_path: Optional[str], artifacts_path: Optional[str]) -> str:
        """Generar reporte completo del experimento"""
        report_dir = self.reports_dir / experiment_id
        report_dir.mkdir(exist_ok=True)
        
        # Obtener datos del experimento
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM experiments WHERE id = ?', (experiment_id,))
        exp_data = cursor.fetchone()
        
        cursor.execute('SELECT * FROM epoch_metrics WHERE experiment_id = ? ORDER BY epoch', (experiment_id,))
        epoch_data = cursor.fetchall()
        
        cursor.execute('SELECT * FROM system_metrics WHERE experiment_id = ? ORDER BY timestamp', (experiment_id,))
        system_data = cursor.fetchall()
        
        conn.close()
        
        # Crear reporte HTML
        report_html = self._create_html_report(exp_data, epoch_data, system_data, metrics, model_path, artifacts_path)
        
        # Guardar reporte
        report_path = report_dir / "report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        # Crear resumen JSON
        summary = {
            "experiment_id": experiment_id,
            "name": exp_data[1] if exp_data else "Unknown",
            "model_type": exp_data[2] if exp_data else "Unknown",
            "category": exp_data[3] if exp_data else "Unknown",
            "duration_seconds": exp_data[6] if exp_data else 0,
            "final_metrics": clean_metrics_for_json(metrics),
            "model_path": model_path,
            "artifacts_path": artifacts_path,
            "epoch_count": len(epoch_data),
            "system_metrics_count": len(system_data),
            "report_generated": datetime.now().isoformat()
        }
        
        summary_path = report_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(report_path)
    
    def _create_html_report(self, exp_data, epoch_data, system_data, metrics, model_path, artifacts_path):
        """Crear reporte HTML detallado"""
        
        # Obtener duración segura
        duration = 0.0
        if exp_data and len(exp_data) > 6 and exp_data[6] is not None:
            try:
                duration = float(exp_data[6])
            except (ValueError, TypeError):
                duration = 0.0
        
        # Formatear métricas de manera segura
        r2_score = metrics.get('r2_score', 0)
        mse = metrics.get('mse', 0)
        mae = metrics.get('mae', 0)
        
        r2_str = f"{r2_score:.4f}" if isinstance(r2_score, (int, float)) and not (np.isnan(r2_score) or np.isinf(r2_score)) else "N/A"
        mse_str = f"{mse:.4f}" if isinstance(mse, (int, float)) and not (np.isnan(mse) or np.isinf(mse)) else "N/A"
        mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) and not (np.isnan(mae) or np.isinf(mae)) else "N/A"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hyperion3 - Reporte de Experimento</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
                .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-left: 4px solid #007bff; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .metric-label {{ font-size: 14px; color: #6c757d; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .status-completed {{ color: #28a745; }}
                .status-error {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Hyperion3 - Reporte de Experimento</h1>
                <p>Experimento: {exp_data[1] if exp_data and len(exp_data) > 1 else 'Unknown'} | ID: {exp_data[0] if exp_data and len(exp_data) > 0 else 'Unknown'}</p>
                <p>Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Métricas Finales</h2>
                <div class="metric">
                    <div class="metric-value">{r2_str}</div>
                    <div class="metric-label">R² Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{mse_str}</div>
                    <div class="metric-label">MSE</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{mae_str}</div>
                    <div class="metric-label">MAE</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{duration:.2f}s</div>
                    <div class="metric-label">Duración</div>
                </div>
            </div>
            
            <div class="section">
                <h2>⚙️ Información del Experimento</h2>
                <table>
                    <tr><th>Campo</th><th>Valor</th></tr>
                    <tr><td>Nombre</td><td>{exp_data[1] if exp_data and len(exp_data) > 1 else 'Unknown'}</td></tr>
                    <tr><td>Tipo de Modelo</td><td>{exp_data[2] if exp_data and len(exp_data) > 2 else 'Unknown'}</td></tr>
                    <tr><td>Categoría</td><td>{exp_data[3] if exp_data and len(exp_data) > 3 else 'Unknown'}</td></tr>
                    <tr><td>Estado</td><td><span class="status-{exp_data[4] if exp_data and len(exp_data) > 4 else 'error'}">{exp_data[4] if exp_data and len(exp_data) > 4 else 'Error'}</span></td></tr>
                    <tr><td>Inicio</td><td>{exp_data[5] if exp_data and len(exp_data) > 5 else 'Unknown'}</td></tr>
                    <tr><td>Fin</td><td>{exp_data[6] if exp_data and len(exp_data) > 6 else 'Unknown'}</td></tr>
                    <tr><td>Modelo Guardado</td><td>{model_path if model_path else 'No'}</td></tr>
                    <tr><td>Artefactos</td><td>{artifacts_path if artifacts_path else 'No'}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>📈 Progreso por Época</h2>
                {self._create_epoch_metrics_table(epoch_data)}
            </div>
            
            <div class="section">
                <h2>💻 Métricas del Sistema</h2>
                {self._create_system_metrics_table(system_data)}
            </div>
        </body>
        </html>
        """
        return html
    
    def _create_epoch_metrics_table(self, epoch_data):
        """Crear tabla de métricas por época"""
        if not epoch_data:
            return "<p>No hay datos de época disponibles.</p>"
        
        html = "<table><tr><th>Época</th><th>Loss</th><th>Val Loss</th><th>R² Score</th><th>Learning Rate</th><th>Timestamp</th></tr>"
        for row in epoch_data[-10:]:  # Últimas 10 épocas
            # Formatear cada valor de forma segura
            epoch = row[2] if len(row) > 2 else 'N/A'
            loss = f"{row[3]:.6f}" if len(row) > 3 and row[3] is not None else 'N/A'
            val_loss = f"{row[4]:.6f}" if len(row) > 4 and row[4] is not None else 'N/A'
            r2_score = f"{row[5]:.4f}" if len(row) > 5 and row[5] is not None else 'N/A'
            lr = f"{row[6]:.6f}" if len(row) > 6 and row[6] is not None else 'N/A'
            timestamp = row[7] if len(row) > 7 else 'N/A'
            
            html += f"<tr><td>{epoch}</td><td>{loss}</td><td>{val_loss}</td><td>{r2_score}</td><td>{lr}</td><td>{timestamp}</td></tr>"
        html += "</table>"
        return html
    
    def _create_system_metrics_table(self, system_data):
        """Crear tabla de métricas del sistema"""
        if not system_data:
            return "<p>No hay datos del sistema disponibles.</p>"
        
        html = "<table><tr><th>CPU %</th><th>Memoria %</th><th>Memoria GB</th><th>GPU Memoria GB</th><th>Timestamp</th></tr>"
        for row in system_data[-5:]:  # Últimas 5 mediciones
            # Formatear cada valor de forma segura
            cpu = f"{row[2]:.1f}%" if len(row) > 2 and row[2] is not None else 'N/A'
            mem_pct = f"{row[3]:.1f}%" if len(row) > 3 and row[3] is not None else 'N/A'
            mem_gb = f"{row[4]:.2f}GB" if len(row) > 4 and row[4] is not None else 'N/A'
            gpu_gb = f"{row[5]:.2f}GB" if len(row) > 5 and row[5] is not None else 'N/A'
            timestamp = row[6] if len(row) > 6 else 'N/A'
            
            html += f"<tr><td>{cpu}</td><td>{mem_pct}</td><td>{mem_gb}</td><td>{gpu_gb}</td><td>{timestamp}</td></tr>"
        html += "</table>"
        return html
    
    def _compare_with_previous_experiments(self, experiment_id: str) -> Dict:
        """Comparar con experimentos anteriores"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Obtener experimento actual
        cursor.execute('SELECT * FROM experiments WHERE id = ?', (experiment_id,))
        current = cursor.fetchone()
        
        if not current:
            conn.close()
            return {}
        
        # Obtener experimentos anteriores del mismo tipo
        cursor.execute('''
            SELECT * FROM experiments 
            WHERE model_type = ? AND category = ? AND id != ? AND status = 'completed'
            ORDER BY r2_score DESC LIMIT 10
        ''', (current[2], current[3], experiment_id))
        
        previous = cursor.fetchall()
        conn.close()
        
        if not previous:
            return {"message": "Primer experimento de este tipo"}
        
        # Comparación
        current_r2 = current[7] if current[7] else 0
        best_previous_r2 = previous[0][7] if previous[0][7] else 0
        
        comparison = {
            "total_previous": len(previous),
            "current_r2": current_r2,
            "best_previous_r2": best_previous_r2,
            "improvement": current_r2 - best_previous_r2,
            "rank": 1,  # Calcular ranking
            "is_best": current_r2 > best_previous_r2
        }
        
        # Calcular ranking
        better_count = sum(1 for exp in previous if exp[7] and exp[7] > current_r2)
        comparison["rank"] = better_count + 1
        
        return comparison
    
    def _show_experiment_summary(self, experiment_id: str, metrics: Dict, comparison: Dict, report_path: str):
        """Mostrar resumen final del experimento"""
        if not self.console or not RICH_AVAILABLE:
            print(f"\n🎯 Experimento {experiment_id} completado")
            r2 = metrics.get('r2_score')
            mse = metrics.get('mse')
            r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) and not (np.isnan(r2) or np.isinf(r2)) else 'N/A'
            mse_str = f"{mse:.4f}" if isinstance(mse, (int, float)) and not (np.isnan(mse) or np.isinf(mse)) else 'N/A'
            print(f"R² Score: {r2_str}")
            print(f"MSE: {mse_str}")
            print(f"Reporte: {report_path}")
            return
        
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.table import Table
        
        # Crear layout rico
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="metrics", size=8),
            Layout(name="comparison", size=6),
            Layout(name="footer", size=3)
        )
        
        # Header
        header = Panel.fit(
            f"[bold green]🎯 EXPERIMENTO COMPLETADO[/bold green]\n"
            f"[cyan]ID: {experiment_id}[/cyan]\n"
            f"[dim]Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="green"
        )
        
        # Métricas
        metrics_table = Table(title="📊 Métricas Finales", show_header=True, header_style="bold magenta")
        metrics_table.add_column("Métrica", style="cyan")
        metrics_table.add_column("Valor", style="green")
        metrics_table.add_column("Estado", style="yellow")
        
        r2_score = metrics.get('r2_score', 0)
        r2_status = "🟢 Excelente" if r2_score > 0.8 else "🟡 Bueno" if r2_score > 0.6 else "🔴 Necesita mejora"
        
        r2_str = f"{r2_score:.4f}" if isinstance(r2_score, (int, float)) and not (np.isnan(r2_score) or np.isinf(r2_score)) else "N/A"
        mse_val = metrics.get('mse', 0)
        mse_str = f"{mse_val:.4f}" if isinstance(mse_val, (int, float)) and not (np.isnan(mse_val) or np.isinf(mse_val)) else "N/A"
        mae_val = metrics.get('mae', 0)
        mae_str = f"{mae_val:.4f}" if isinstance(mae_val, (int, float)) and not (np.isnan(mae_val) or np.isinf(mae_val)) else "N/A"
        
        metrics_table.add_row("R² Score", r2_str, r2_status)
        metrics_table.add_row("MSE", mse_str, "📊 Loss")
        metrics_table.add_row("MAE", mae_str, "📊 Error Abs")
        
        # Comparación
        if comparison:
            comp_table = Table(title="📈 Comparación con Experimentos Anteriores", show_header=True)
            comp_table.add_column("Aspecto", style="cyan")
            comp_table.add_column("Valor", style="white")
            
            if comparison.get('is_best'):
                comp_table.add_row("🏆 Estado", "[bold green]¡NUEVO MEJOR MODELO![/bold green]")
            else:
                comp_table.add_row("📊 Ranking", f"#{comparison.get('rank', 'N/A')} de {comparison.get('total_previous', 0) + 1}")
            
            improvement = comparison.get('improvement', 0)
            if improvement > 0:
                imp_str = f"{improvement:.4f}" if isinstance(improvement, (int, float)) and not (np.isnan(improvement) or np.isinf(improvement)) else "N/A"
                comp_table.add_row("📈 Mejora R²", f"[green]+{imp_str}[/green]")
            else:
                imp_str = f"{improvement:.4f}" if isinstance(improvement, (int, float)) and not (np.isnan(improvement) or np.isinf(improvement)) else "N/A"
                comp_table.add_row("📉 Diferencia R²", f"[red]{imp_str}[/red]")
        else:
            comp_table = Panel("[yellow]Primer experimento de este tipo[/yellow]", title="📈 Comparación")
        
        # Footer
        footer = Panel(
            f"[dim]💾 Reporte completo: {report_path}[/dim]\n"
            f"[dim]📁 Artefactos guardados en mlops/[/dim]",
            border_style="dim"
        )
        
        layout["header"].update(header)
        layout["metrics"].update(metrics_table)
        layout["comparison"].update(comp_table)
        layout["footer"].update(footer)
        
        self.console.print(layout)
    
    def _log_experiment_event(self, message: str, data: Optional[Dict] = None):
        """Registrar evento del experimento"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "message": message,
            "data": clean_metrics_for_json(data or {})
        }
        
        log_file = self.logs_dir / f"{datetime.now().strftime('%Y%m%d')}_experiments.log"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def get_experiments_summary(self) -> Dict:
        """Obtener resumen de todos los experimentos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM experiments')
        total_experiments = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM experiments WHERE status = "completed"')
        completed_experiments = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(r2_score) FROM experiments WHERE status = "completed" AND r2_score IS NOT NULL')
        avg_r2 = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT MAX(r2_score) FROM experiments WHERE status = "completed" AND r2_score IS NOT NULL')
        best_r2 = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT * FROM experiments WHERE status = "completed" ORDER BY r2_score DESC LIMIT 5')
        top_experiments = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_experiments": total_experiments,
            "completed_experiments": completed_experiments,
            "success_rate": completed_experiments / total_experiments if total_experiments > 0 else 0,
            "average_r2": avg_r2,
            "best_r2": best_r2,
            "top_experiments": top_experiments
        }
    
    def create_training_progress_monitor(self, experiment_id: str, total_epochs: int = 100):
        """Crear monitor de progreso para entrenamiento"""
        if not RICH_AVAILABLE:
            return SimpleProgressMonitor(experiment_id, total_epochs)
        
        return RichProgressMonitor(self.console, experiment_id, total_epochs, self)

class SimpleProgressMonitor:
    """Monitor simple sin Rich"""
    
    def __init__(self, experiment_id: str, total_epochs: int):
        self.experiment_id = experiment_id
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def update(self, epoch: int, metrics: Dict):
        self.current_epoch = epoch
        progress = (epoch / self.total_epochs) * 100
        loss = metrics.get('loss')
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) and not (np.isnan(loss) or np.isinf(loss)) else 'N/A'
        print(f"\rÉpoca {epoch}/{self.total_epochs} ({progress:.1f}%) - Loss: {loss_str}", end='')
    
    def finish(self):
        print("\n✅ Entrenamiento completado")

class RichProgressMonitor:
    """Monitor avanzado con Rich"""
    
    def __init__(self, console, experiment_id: str, total_epochs: int, mlops_manager):
        self.console = console
        self.experiment_id = experiment_id
        self.total_epochs = total_epochs
        self.mlops_manager = mlops_manager
        self.progress = None
        self.task_id = None
        
    def __enter__(self):
        if not RICH_AVAILABLE:
            return self
            
        # Usar directamente RichProgress con protección manual
        from rich.progress import Progress as RichProgress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
        
        # Verificar si ya hay un progress activo y esperar
        import time
        max_wait = 5  # máximo 5 segundos de espera
        waited = 0
        while hasattr(RichProgress, '_live') and getattr(RichProgress, '_live', None) is not None and waited < max_wait:
            time.sleep(0.1)
            waited += 0.1
        
        self.progress = RichProgress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
        self.progress.__enter__()
        self.task_id = self.progress.add_task("Entrenando...", total=self.total_epochs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, epoch: int, metrics: Dict):
        if self.progress and self.task_id is not None:
            # Actualizar progress bar
            loss = metrics.get('loss')
            r2 = metrics.get('r2_score')
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) and not (np.isnan(loss) or np.isinf(loss)) else 'N/A'
            r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) and not (np.isnan(r2) or np.isinf(r2)) else 'N/A'
            self.progress.update(
                self.task_id, 
                completed=epoch, 
                description=f"Época {epoch}/{self.total_epochs} - Loss: {loss_str} - R²: {r2_str}"
            )
            
            # Registrar métricas
            self.mlops_manager.log_epoch_metrics(self.experiment_id, epoch, metrics)
            
            # Registrar métricas del sistema cada 10 épocas
            if epoch % 10 == 0:
                self.mlops_manager.log_system_metrics(self.experiment_id)
    
    def finish(self):
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, completed=self.total_epochs, description="✅ Entrenamiento completado")
