# Instrucciones de instalación y ejecución

Esta guía reproduce el Proyecto 1 desde la raíz del repositorio. Los comandos están escritos para Windows PowerShell y mantienen el entorno virtual dentro de `configuracion/`, de modo que la raíz continúe ordenada.

## 1. Requisitos previos

- Windows 10 u 11 de 64 bits.
- Python 3.13.1 de 64 bits. Las versiones fijadas fueron comprobadas con esta versión.
- Git para clonar el repositorio.
- Cuenta de Kaggle con las reglas de IEEE-CIS Fraud Detection aceptadas.
- Conexión a Internet para instalar paquetes y descargar los datos.
- Se recomiendan 16 GB de RAM y al menos 3 GB libres en disco.
- TinyTeX, TeX Live o MiKTeX únicamente si se desea recompilar `informe.pdf`.
- Google Chrome o Microsoft Edge únicamente si se desea volver a imprimir la presentación HTML como PDF.

La ejecución principal funciona en CPU. No es obligatorio disponer de GPU.

## 2. Obtener el repositorio

```powershell
git clone https://github.com/DanielBarillasM/Proyecto-1_Grupo-1_DLYSI_Sec-30.git
Set-Location Proyecto-1_Grupo-1_DLYSI_Sec-30
```

Todos los comandos posteriores deben ejecutarse desde esa raíz.

## 3. Crear y activar el entorno virtual

Compruebe primero la versión de Python:

```powershell
python --version
```

El resultado esperado es `Python 3.13.1`. Después cree el entorno dentro de la carpeta de configuración:

```powershell
python -m venv configuracion/.venv
.\configuracion\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Si PowerShell impide activar el entorno, habilite scripts únicamente para la sesión actual y repita la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\configuracion\.venv\Scripts\Activate.ps1
```

## 4. Instalar las dependencias

```powershell
python -m pip install -r configuracion/requirements.txt
```

Compruebe las bibliotecas principales:

```powershell
python -c "import torch, sklearn, pandas, kagglehub; print('PyTorch', torch.__version__); print('scikit-learn', sklearn.__version__); print('pandas', pandas.__version__)"
```

Las versiones están fijadas en `requirements.txt` para impedir que una actualización posterior cambie silenciosamente los resultados.

## 5. Configurar Kaggle de forma segura

Antes de descargar, abra la competencia [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) en el navegador y acepte sus reglas. Después ejecute:

```powershell
python -c "import kagglehub; kagglehub.login()"
```

Complete el inicio de sesión solicitado por Kaggle. No copie tokens, claves ni archivos de credenciales dentro del repositorio.

## 6. Descargar los datos

```powershell
python codigo/download_data.py
```

El script descarga y extrae:

```text
datos/raw/train_transaction.csv
datos/raw/train_identity.csv
```

Los CSV están excluidos de Git y no deben subirse al repositorio.

## 7. Elegir el tipo de ejecución

### Opción A: revisar la entrega existente

Los modelos, resultados, figuras y entregables ya están ejecutados. Para comprobarlos sin entrenar nuevamente:

```powershell
python codigo/audit_project1.py
```

La auditoría debe informar:

- cero errores en el notebook;
- informe de cuatro páginas;
- presentación de ocho diapositivas;
- candidato `A`;
- cero archivos sueltos en la raíz;
- ficha DOCX válida.

### Opción B: reproducir todo el experimento

Advertencia: el comando siguiente usa `force=True`, vuelve a entrenar A, B y C, recalcula umbrales y reemplaza los artefactos existentes. En CPU puede tardar varios minutos.

```powershell
python codigo/proyecto1_pipeline.py
```

Las salidas se guardan en:

```text
artefactos/
evidencia/figuras/
```

La semilla principal es 2026. El pipeline conserva el corte cronológico, ajusta el preprocesamiento con entrenamiento y abre prueba después de congelar candidato y umbrales.

## 8. Regenerar los documentos fuente

Después de reproducir los modelos, regenere el notebook, el informe LaTeX, la presentación HTML, el README, el manifiesto y este archivo de dependencias:

```powershell
python codigo/build_deliverables.py
```

Ejecute el notebook y guarde sus salidas:

```powershell
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 entregables/cuaderno/proyecto1_calderon_barillas.ipynb
```

Para abrirlo interactivamente:

```powershell
jupyter lab entregables/cuaderno/proyecto1_calderon_barillas.ipynb
```

## 9. Compilar el informe PDF

Este paso requiere que `pdflatex` esté disponible en `PATH`:

```powershell
Push-Location entregables/informe
pdflatex -interaction=nonstopmode -halt-on-error informe.tex
pdflatex -interaction=nonstopmode -halt-on-error informe.tex
Pop-Location
```

Se ejecuta dos veces para resolver correctamente referencias y metadatos internos.

## 10. Generar la ficha DOCX

```powershell
python codigo/crear_ficha_repositorio.py
```

La salida se guarda en `entregables/ficha/Ficha_Repositorio_Proyecto1.docx`. El documento incluye un QR que apunta al repositorio.

## 11. Presentación HTML y PDF

Abra la presentación interactiva con:

```powershell
Start-Process entregables/presentacion/presentacion.html
```

Use las flechas, barra espaciadora o teclas `Page Up` y `Page Down` para navegar. Para recrear el PDF, abra el HTML en Chrome o Edge, seleccione **Imprimir**, elija **Guardar como PDF**, orientación horizontal, márgenes ninguno y gráficos de fondo activados. El resultado debe contener exactamente ocho páginas.

## 12. Auditoría final

```powershell
python codigo/audit_project1.py
```

La auditoría valida rutas, artefactos, ejecución del notebook, estructura HTML, número de páginas, falsificaciones, ficha DOCX y organización de la raíz.

## 13. Solución de problemas

### Kaggle responde 401 o 403

- Confirme que aceptó las reglas de la competencia.
- Ejecute nuevamente `kagglehub.login()`.
- Verifique que la cuenta activa sea la autorizada.

### No se encuentran los CSV

Ejecute `python codigo/download_data.py` y compruebe que ambos archivos estén en `datos/raw/`.

### PowerShell bloquea `Activate.ps1`

Use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. La modificación dura solamente durante la terminal actual.

### `pdflatex` no se reconoce

Instale TinyTeX, TeX Live o MiKTeX y abra una terminal nueva. Este componente no se instala mediante `pip`.

### Aparece una advertencia de ZMQ en Windows

La advertencia relacionada con `Proactor event loop` puede aparecer al ejecutar `nbconvert`. Si el proceso finaliza con código 0 y no existen salidas de tipo `error`, no invalida el notebook.

### PyTorch no se instala desde el índice predeterminado

Consulte el selector oficial de instalación de PyTorch para su plataforma. Instale la variante CPU o CUDA compatible y vuelva a ejecutar la instalación del resto de dependencias.

## 14. Finalizar la sesión

```powershell
deactivate
```

El entorno virtual permanece en `configuracion/.venv/` y está excluido del control de versiones.
