"""
QUANTUM DREAM ATLAS
The Most Aesthetically Beautiful & Scientifically Advanced Visualization Engine
21 Advanced Subjects × 16 Dynamic Plots = 336 Nobel-Tier Visualizations
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.linalg as la
from scipy.spatial import Delaunay
from scipy.stats import multivariate_normal
from scipy.integrate import odeint
from scipy.special import sph_harm_y, legendre
import colorsys

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Quantum Dream Atlas ✨",
    layout="wide",
    page_icon="✨",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
    }
    div.block-container {
        padding-top: 2rem;
    }
    h1, h2, h3, h4 {
        color: #00ffaa !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# MASTER RANDOM SEED GENERATOR
# ============================================================
if 'master_seed' not in st.session_state:
    st.session_state.master_seed = np.random.randint(0, 1000000)

def get_seed(offset=0):
    """Generate deterministic but unique seed for each plot"""
    return st.session_state.master_seed + offset

# ============================================================
# ADVANCED PLOTTING FUNCTIONS (Perspective Variations)
# ============================================================

def create_dark_layout():
    """Master dark theme layout"""
    return dict(
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=dict(color='#00ffaa', size=10),
        scene=dict(
            xaxis=dict(backgroundcolor='#000000', gridcolor='#001a1a', showbackground=True),
            yaxis=dict(backgroundcolor='#000000', gridcolor='#001a1a', showbackground=True),
            zaxis=dict(backgroundcolor='#000000', gridcolor='#001a1a', showbackground=True)
        ),
        xaxis=dict(gridcolor='#001a1a', zerolinecolor='#003333'),
        yaxis=dict(gridcolor='#001a1a', zerolinecolor='#003333'),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        height=400
    )

def quantum_colorscale(name='viridis'):
    """Beautiful dark-themed colorscales"""
    scales = {
        'viridis': [[0, '#000033'], [0.3, '#440154'], [0.6, '#31688e'], [0.8, '#35b779'], [1, '#fde724']],
        'plasma': [[0, '#0d0887'], [0.3, '#7e03a8'], [0.6, '#cc4778'], [0.8, '#f89540'], [1, '#f0f921']],
        'inferno': [[0, '#000004'], [0.3, '#420a68'], [0.6, '#932667'], [0.8, '#dd513a'], [1, '#fcffa4']],
        'magma': [[0, '#000004'], [0.3, '#3b0f70'], [0.6, '#8c2981'], [0.8, '#de4968'], [1, '#fcfdbf']],
        'turbo': [[0, '#30123b'], [0.25, '#4777ef'], [0.5, '#1ac938'], [0.75, '#f0b32f'], [1, '#7a0403']],
        'quantum': [[0, '#000033'], [0.2, '#0a4d8c'], [0.4, '#00a5a8'], [0.6, '#00d696'], [0.8, '#7ef542'], [1, '#f0ff00']],
        'dream': [[0, '#000000'], [0.25, '#1a0033'], [0.5, '#4d0099'], [0.75, '#9933ff'], [1, '#00ffff']],
        'fire': [[0, '#000000'], [0.25, '#330000'], [0.5, '#cc0000'], [0.75, '#ff6600'], [1, '#ffff00']],
        'ice': [[0, '#000033'], [0.25, '#003366'], [0.5, '#0066cc'], [0.75, '#33ccff'], [1, '#ffffff']],
        'aurora': [[0, '#001a00'], [0.3, '#004d00'], [0.5, '#00ff00'], [0.7, '#00ffcc'], [1, '#00ffff']],
    }
    return scales.get(name, scales['quantum'])

# ============================================================
# CORE VISUALIZATION ENGINE
# ============================================================

def manifold_3d(seed, subject_id, plot_id, colorscale='quantum'):
    """3D manifold surface - base template for all subjects"""
    np.random.seed(seed)
    n = 80
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)
    
    # Subject-specific mathematical transformation
    phi = subject_id * 0.314
    psi = plot_id * 0.271
    
    Z = (np.sin(X * np.cos(phi) + Y * np.sin(phi)) * 
         np.exp(-0.1 * (X**2 + Y**2)) * 
         np.cos(X * Y * psi) * 
         (1 + 0.3 * np.sin(X * 3 + phi) * np.cos(Y * 3 + psi)))
    
    Z += 0.2 * np.random.randn(n, n) * np.exp(-0.2 * (X**2 + Y**2))
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale=quantum_colorscale(colorscale),
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
        lightposition=dict(x=100, y=100, z=200)
    )])
    
    fig.update_layout(**create_dark_layout())
    fig.update_layout(
        scene=dict(
            camera=dict(eye=dict(x=1.5*np.cos(phi), y=1.5*np.sin(psi), z=1.2)),
            aspectmode='cube'
        )
    )
    return fig

def quantum_field_2d(seed, subject_id, plot_id, colorscale='viridis'):
    """2D quantum field heatmap"""
    np.random.seed(seed)
    n = 120
    x = np.linspace(-4, 4, n)
    y = np.linspace(-4, 4, n)
    X, Y = np.meshgrid(x, y)
    
    phi = subject_id * 0.314
    psi = plot_id * 0.271
    
    Z = np.exp(-0.15 * (X**2 + Y**2)) * (
        np.sin(X * 2 * np.cos(phi) + Y * 2 * np.sin(psi)) +
        0.5 * np.cos(X * Y * 3) * np.sin((X**2 + Y**2) * psi)
    )
    
    fig = go.Figure(data=go.Heatmap(
        x=x, y=y, z=Z,
        colorscale=quantum_colorscale(colorscale),
        showscale=False
    ))
    
    fig.update_layout(**create_dark_layout())
    return fig

def topology_network(seed, subject_id, plot_id):
    """Topological network visualization"""
    np.random.seed(seed)
    n = 50
    
    # Generate points in topological space
    theta = np.linspace(0, 2*np.pi, n)
    r = 1 + 0.3 * np.sin(subject_id * theta + plot_id)
    x = r * np.cos(theta) + 0.1 * np.random.randn(n)
    y = r * np.sin(theta) + 0.1 * np.random.randn(n)
    z = 0.5 * np.sin(3 * theta + plot_id) + 0.1 * np.random.randn(n)
    
    # Compute connections based on distance threshold
    threshold = 0.6
    edges_x, edges_y, edges_z = [], [], []
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)
            if dist < threshold:
                edges_x.extend([x[i], x[j], None])
                edges_y.extend([y[i], y[j], None])
                edges_z.extend([z[i], z[j], None])
    
    fig = go.Figure()
    
    # Edges
    fig.add_trace(go.Scatter3d(
        x=edges_x, y=edges_y, z=edges_z,
        mode='lines',
        line=dict(color='#00ffaa', width=1.5),
        opacity=0.3,
        hoverinfo='none'
    ))
    
    # Nodes
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=4, color=z, colorscale=quantum_colorscale('quantum'), showscale=False),
        hoverinfo='none'
    ))
    
    fig.update_layout(**create_dark_layout())
    fig.update_layout(scene=dict(aspectmode='cube'))
    return fig

def phase_space_flow(seed, subject_id, plot_id):
    """Phase space flow field"""
    np.random.seed(seed)
    n = 20
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)
    
    phi = subject_id * 0.314
    psi = plot_id * 0.271
    
    U = -Y + X * np.sin(phi)
    V = X + Y * np.cos(psi)
    
    # Normalize for better visualization
    M = np.sqrt(U**2 + V**2)
    U = U / (M + 0.1)
    V = V / (M + 0.1)
    
    fig = go.Figure()
    
    # Background potential
    Z = np.exp(-0.1 * (X**2 + Y**2)) * np.sin(X * Y + phi)
    fig.add_trace(go.Heatmap(
        x=x, y=y, z=Z,
        colorscale=quantum_colorscale('dream'),
        showscale=False,
        opacity=0.5
    ))
    
    # Vector field
    for i in range(0, n, 2):
        for j in range(0, n, 2):
            fig.add_trace(go.Scatter(
                x=[x[i], x[i] + U[j, i]*0.2],
                y=[y[j], y[j] + V[j, i]*0.2],
                mode='lines',
                line=dict(color='#00ffaa', width=2),
                hoverinfo='none',
                showlegend=False
            ))
    
    fig.update_layout(**create_dark_layout())
    return fig

def entropy_landscape(seed, subject_id, plot_id):
    """Entropy/information landscape"""
    np.random.seed(seed)
    n = 80
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)
    
    phi = subject_id * 0.314
    psi = plot_id * 0.271
    
    # Information-theoretic potential
    R2 = X**2 + Y**2
    Z = -np.log(np.exp(-R2) + 0.1) + np.sin(X * 2 + phi) * np.cos(Y * 2 + psi)
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale=quantum_colorscale('inferno'),
        showscale=False,
        contours=dict(
            z=dict(show=True, color='#00ffaa', width=1, highlightwidth=2)
        )
    )])
    
    fig.update_layout(**create_dark_layout())
    fig.update_layout(scene=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))))
    return fig

def attractor_dynamics(seed, subject_id, plot_id):
    """Strange attractor dynamics"""
    np.random.seed(seed)
    
    # Lorenz-like system with subject-specific parameters
    sigma = 10 + subject_id * 0.5
    rho = 28 + plot_id * 0.3
    beta = 8/3
    
    def lorenz(state, t):
        x, y, z = state
        return [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ]
    
    t = np.linspace(0, 20, 2000)
    state0 = [1.0, 1.0, 1.0]
    states = odeint(lorenz, state0, t)
    
    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(
            color=np.linspace(0, 1, len(x)),
            colorscale=quantum_colorscale('turbo'),
            width=2
        ),
        hoverinfo='none'
    )])
    
    fig.update_layout(**create_dark_layout())
    fig.update_layout(scene=dict(aspectmode='cube'))
    return fig

def correlation_matrix_3d(seed, subject_id, plot_id):
    """3D correlation/covariance structure"""
    np.random.seed(seed)
    n = 30
    
    # Generate correlation matrix
    A = np.random.randn(n, n) * (1 + subject_id * 0.1)
    C = A @ A.T
    C = C / np.max(np.abs(C))
    
    # Add structure
    for i in range(n):
        for j in range(n):
            C[i, j] *= np.exp(-0.1 * abs(i - j) * (1 + 0.1 * plot_id))
    
    x, y = np.meshgrid(range(n), range(n))
    
    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=C,
        colorscale=quantum_colorscale('plasma'),
        showscale=False
    )])
    
    fig.update_layout(**create_dark_layout())
    return fig

def wave_interference(seed, subject_id, plot_id):
    """Quantum wave interference pattern"""
    np.random.seed(seed)
    n = 150
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    X, Y = np.meshgrid(x, y)
    
    # Multiple wave sources
    k = 2 * np.pi + subject_id * 0.2
    sources = [
        (1 * np.cos(plot_id), 1 * np.sin(plot_id)),
        (-1 * np.cos(plot_id), -1 * np.sin(plot_id)),
        (0, 0)
    ]
    
    Z = np.zeros_like(X)
    for sx, sy in sources:
        R = np.sqrt((X - sx)**2 + (Y - sy)**2)
        Z += np.cos(k * R) * np.exp(-0.1 * R)
    
    fig = go.Figure(data=go.Heatmap(
        x=x, y=y, z=Z,
        colorscale=quantum_colorscale('ice'),
        showscale=False
    ))
    
    fig.update_layout(**create_dark_layout())
    return fig

def gradient_flow_3d(seed, subject_id, plot_id):
    """Gradient descent flow on manifold"""
    np.random.seed(seed)
    n = 60
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)
    
    phi = subject_id * 0.314
    psi = plot_id * 0.271
    
    # Potential surface
    Z = (X**2 + Y**2) * 0.5 + np.sin(X * 3 + phi) * 0.3 + np.cos(Y * 3 + psi) * 0.3
    
    # Gradient computation
    dZdx = np.gradient(Z, axis=1)
    dZdy = np.gradient(Z, axis=0)
    
    fig = go.Figure()
    
    # Surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=quantum_colorscale('magma'),
        showscale=False,
        opacity=0.8
    ))
    
    # Gradient paths
    for _ in range(10):
        x0, y0 = np.random.uniform(-1.5, 1.5, 2)
        path_x, path_y, path_z = [x0], [y0], []
        
        for step in range(50):
            ix = int((x0 + 2) * n / 4)
            iy = int((y0 + 2) * n / 4)
            if 0 <= ix < n and 0 <= iy < n:
                path_z.append(Z[iy, ix])
                dx = -dZdx[iy, ix] * 0.05
                dy = -dZdy[iy, ix] * 0.05
                x0 += dx
                y0 += dy
                path_x.append(x0)
                path_y.append(y0)
            else:
                break
        
        if len(path_z) > 2:
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines',
                line=dict(color='#00ffaa', width=3),
                showlegend=False,
                hoverinfo='none'
            ))
    
    fig.update_layout(**create_dark_layout())
    return fig

def spherical_harmonics(seed, subject_id, plot_id):
    """Spherical harmonics visualization"""
    np.random.seed(seed)
    
    l = (subject_id % 5) + 1
    m = ((plot_id * subject_id) % (2 * l + 1)) - l
    
    theta = np.linspace(0, np.pi, 80)
    phi = np.linspace(0, 2*np.pi, 80)
    THETA, PHI = np.meshgrid(theta, phi)
    
    Y = sph_harm(m, l, PHI, THETA)
    R = np.abs(Y)
    
    X = R * np.sin(THETA) * np.cos(PHI)
    Y_coord = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y_coord, z=Z,
        surfacecolor=np.angle(Y),
        colorscale=quantum_colorscale('aurora'),
        showscale=False
    )])
    
    fig.update_layout(**create_dark_layout())
    return fig

def tensor_field(seed, subject_id, plot_id):
    """Tensor field visualization"""
    np.random.seed(seed)
    n = 15
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)
    
    phi = subject_id * 0.314
    psi = plot_id * 0.271
    
    # Tensor components
    T11 = np.cos(X + phi) * np.sin(Y + psi)
    T12 = np.sin(X + phi) * np.cos(Y + psi)
    T22 = np.cos(X * Y + phi + psi)
    
    fig = go.Figure()
    
    # Background field
    trace_T = T11 + T22
    fig.add_trace(go.Heatmap(
        x=x, y=y, z=trace_T,
        colorscale=quantum_colorscale('dream'),
        showscale=False,
        opacity=0.4
    ))
    
    # Tensor ellipses
    for i in range(0, n, 2):
        for j in range(0, n, 2):
            T = np.array([[T11[j, i], T12[j, i]], [T12[j, i], T22[j, i]]])
            eigenvalues, eigenvectors = la.eigh(T)
            
            angle = np.linspace(0, 2*np.pi, 30)
            ellipse = eigenvectors @ np.diag(np.abs(eigenvalues)) @ np.array([np.cos(angle), np.sin(angle)])
            
            fig.add_trace(go.Scatter(
                x=x[i] + ellipse[0] * 0.1,
                y=y[j] + ellipse[1] * 0.1,
                mode='lines',
                line=dict(color='#00ffaa', width=1.5),
                fill='toself',
                fillcolor='rgba(0, 255, 170, 0.1)',
                showlegend=False,
                hoverinfo='none'
            ))
    
    fig.update_layout(**create_dark_layout())
    return fig

def bifurcation_diagram(seed, subject_id, plot_id):
    """Bifurcation cascade"""
    np.random.seed(seed)
    
    r_values = np.linspace(2.8 + subject_id * 0.05, 4.0, 300)
    iterations = 200
    last_n = 100
    
    x = []
    r = []
    
    for r_val in r_values:
        x_val = 0.1 + np.random.rand() * 0.01
        for _ in range(iterations):
            x_val = r_val * x_val * (1 - x_val) * (1 + 0.01 * np.sin(plot_id))
        
        for _ in range(last_n):
            x_val = r_val * x_val * (1 - x_val) * (1 + 0.01 * np.sin(plot_id))
            x.append(x_val)
            r.append(r_val)
    
    fig = go.Figure(data=go.Scattergl(
        x=r, y=x,
        mode='markers',
        marker=dict(
            size=0.5,
            color=x,
            colorscale=quantum_colorscale('turbo'),
            showscale=False
        ),
        hoverinfo='none'
    ))
    
    fig.update_layout(**create_dark_layout())
    fig.update_xaxes(title_text='Parameter r')
    fig.update_yaxes(title_text='x')
    return fig

def quantum_tunneling(seed, subject_id, plot_id):
    """Quantum tunneling wavefunction"""
    np.random.seed(seed)
    x = np.linspace(-5, 5, 200)
    
    # Potential barrier
    V = np.zeros_like(x)
    barrier_height = 2 + subject_id * 0.1
    barrier_width = 1 + plot_id * 0.05
    V[(x > -barrier_width/2) & (x < barrier_width/2)] = barrier_height
    
    # Wavefunction (approximate)
    E = barrier_height * 0.7
    k1 = np.sqrt(2 * E)
    k2 = np.sqrt(2 * (barrier_height - E)) if E < barrier_height else 1j * np.sqrt(2 * (E - barrier_height))
    
    psi = np.zeros_like(x, dtype=complex)
    psi[x < -barrier_width/2] = np.exp(1j * k1 * x[x < -barrier_width/2])
    psi[(x >= -barrier_width/2) & (x <= barrier_width/2)] = np.exp(-np.abs(k2) * x[(x >= -barrier_width/2) & (x <= barrier_width/2)])
    psi[x > barrier_width/2] = 0.5 * np.exp(1j * k1 * x[x > barrier_width/2])
    
    fig = go.Figure()
    
    # Potential
    fig.add_trace(go.Scatter(
        x=x, y=V,
        mode='lines',
        line=dict(color='#ff0066', width=2),
        name='Potential',
        fill='tozeroy',
        fillcolor='rgba(255, 0, 102, 0.2)'
    ))
    
    # Wavefunction
    fig.add_trace(go.Scatter(
        x=x, y=np.abs(psi)**2 * 2,
        mode='lines',
        line=dict(color='#00ffaa', width=2),
        name='|ψ|²',
        fill='tozeroy',
        fillcolor='rgba(0, 255, 170, 0.2)'
    ))
    
    fig.update_layout(**create_dark_layout())
    return fig

def fractal_dimension(seed, subject_id, plot_id):
    """Fractal structure visualization"""
    np.random.seed(seed)
    
    # Julia set variant
    n = 400
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    c = complex(-0.7 + subject_id * 0.05, 0.27 + plot_id * 0.05)
    
    iterations = np.zeros_like(Z, dtype=int)
    mask = np.ones_like(Z, dtype=bool)
    
    for i in range(50):
        Z[mask] = Z[mask]**2 + c
        mask = np.abs(Z) < 2
        iterations[~mask & (iterations == 0)] = i
    
    fig = go.Figure(data=go.Heatmap(
        x=x, y=y, z=iterations,
        colorscale=quantum_colorscale('turbo'),
        showscale=False
    ))
    
    fig.update_layout(**create_dark_layout())
    return fig

def gauge_field(seed, subject_id, plot_id):
    """Gauge field configuration"""
    np.random.seed(seed)
    n = 20
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)
    
    phi = subject_id * 0.314
    psi = plot_id * 0.271
    
    # Gauge potential
    Ax = Y * np.exp(-(X**2 + Y**2) * 0.5)
    Ay = -X * np.exp(-(X**2 + Y**2) * 0.5)
    
    # Field strength
    F = np.gradient(Ay, axis=1) - np.gradient(Ax, axis=0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        x=x, y=y, z=F,
        colorscale=quantum_colorscale('fire'),
        showscale=False,
        opacity=0.6
    ))
    
    # Vector field
    for i in range(0, n, 2):
        for j in range(0, n, 2):
            fig.add_annotation(
                x=x[i], y=y[j],
                ax=x[i] + Ax[j, i] * 0.3,
                ay=y[j] + Ay[j, i] * 0.3,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#00ffaa'
            )
    
    fig.update_layout(**create_dark_layout())
    return fig

def eigenspace_projection(seed, subject_id, plot_id):
    """Eigenspace projection (PCA-like)"""
    np.random.seed(seed)
    
    # Generate high-dimensional data
    n_samples = 300
    n_dims = 20
    
    # Create structured covariance
    cov = np.eye(n_dims)
    for i in range(n_dims):
        for j in range(n_dims):
            cov[i, j] = np.exp(-0.5 * abs(i - j)) * np.cos(subject_id * 0.1 * (i + j))
    
    data = np.random.multivariate_normal(np.zeros(n_dims), cov, n_samples)
    
    # Compute eigenvectors
    cov_matrix = np.cov(data.T)
    eigenvalues, eigenvectors = la.eigh(cov_matrix)
    
    # Project onto top 3 components
    idx = np.argsort(eigenvalues)[::-1]
    top3 = eigenvectors[:, idx[:3]]
    projected = data @ top3
    
    colors = np.sqrt(projected[:, 0]**2 + projected[:, 1]**2 + projected[:, 2]**2)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=projected[:, 0],
        y=projected[:, 1],
        z=projected[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            colorscale=quantum_colorscale('quantum'),
            showscale=False,
            opacity=0.7
        ),
        hoverinfo='none'
    )])
    
    fig.update_layout(**create_dark_layout())
    return fig

# ============================================================
# SUBJECT DEFINITIONS
# ============================================================

SUBJECTS = {
    'Reinforcement Learning': {
        'icon': '🎮',
        'plots': [
            ('Q-Value Landscape', manifold_3d, 'viridis'),
            ('Policy Gradient Flow', phase_space_flow, None),
            ('Value Function Surface', entropy_landscape, None),
            ('State-Action Space', quantum_field_2d, 'plasma'),
            ('Reward Manifold', manifold_3d, 'turbo'),
            ('Bellman Error Field', wave_interference, None),
            ('Advantage Function', gradient_flow_3d, None),
            ('Experience Replay Network', topology_network, None),
            ('Temporal Difference', correlation_matrix_3d, None),
            ('Actor-Critic Dynamics', attractor_dynamics, None),
            ('Exploration Landscape', entropy_landscape, None),
            ('Multi-Agent Topology', topology_network, None),
            ('Return Distribution', bifurcation_diagram, None),
            ('Credit Assignment', tensor_field, None),
            ('Eigenoptions', eigenspace_projection, None),
            ('Meta-Learning Manifold', manifold_3d, 'quantum'),
        ]
    },
    
    'Artificial General Intelligence': {
        'icon': '🧠',
        'plots': [
            ('AGI Capability Manifold', manifold_3d, 'dream'),
            ('Cognitive Architecture', topology_network, None),
            ('Transfer Learning Field', quantum_field_2d, 'aurora'),
            ('Multi-Task Gradient', gradient_flow_3d, None),
            ('World Model Latent Space', eigenspace_projection, None),
            ('Reasoning Dynamics', attractor_dynamics, None),
            ('Knowledge Graph Topology', topology_network, None),
            ('Abstraction Hierarchy', entropy_landscape, None),
            ('Meta-Cognition Surface', manifold_3d, 'magma'),
            ('Goal Alignment Field', phase_space_flow, None),
            ('Causal Discovery Network', topology_network, None),
            ('Attention Manifold', correlation_matrix_3d, None),
            ('Planning Landscape', wave_interference, None),
            ('Few-Shot Learning', bifurcation_diagram, None),
            ('Emergent Behavior', fractal_dimension, None),
            ('Universal Intelligence', manifold_3d, 'turbo'),
        ]
    },
    
    'Quantum Physics': {
        'icon': '⚛️',
        'plots': [
            ('Wavefunction Manifold', manifold_3d, 'quantum'),
            ('Quantum Superposition', wave_interference, None),
            ('Hilbert Space Projection', eigenspace_projection, None),
            ('Energy Eigenstate', spherical_harmonics, None),
            ('Quantum Tunneling', quantum_tunneling, None),
            ('Entanglement Entropy', entropy_landscape, None),
            ('Phase Space Density', phase_space_flow, None),
            ('Observable Operator', tensor_field, None),
            ('Coherence Landscape', quantum_field_2d, 'ice'),
            ('Decoherence Dynamics', attractor_dynamics, None),
            ('Quantum Chaos', bifurcation_diagram, None),
            ('Measurement Manifold', manifold_3d, 'plasma'),
            ('Spin Network', topology_network, None),
            ('Quantum Correlations', correlation_matrix_3d, None),
            ('Uncertainty Surface', entropy_landscape, None),
            ('Many-Body State', fractal_dimension, None),
        ]
    },
    
    'Topology': {
        'icon': '🔗',
        'plots': [
            ('Topological Manifold', manifold_3d, 'turbo'),
            ('Homology Groups', topology_network, None),
            ('Betti Number Flow', phase_space_flow, None),
            ('Persistent Homology', bifurcation_diagram, None),
            ('Simplicial Complex', topology_network, None),
            ('Morse Theory Landscape', entropy_landscape, None),
            ('Fiber Bundle', gradient_flow_3d, None),
            ('Knot Invariant', attractor_dynamics, None),
            ('Homotopy Groups', quantum_field_2d, 'aurora'),
            ('Manifold Curvature', tensor_field, None),
            ('Topological Defects', wave_interference, None),
            ('Cobordism Class', manifold_3d, 'dream'),
            ('Euler Characteristic', correlation_matrix_3d, None),
            ('Genus Landscape', entropy_landscape, None),
            ('Bundle Connection', gauge_field, None),
            ('Characteristic Classes', eigenspace_projection, None),
        ]
    },
    
    'Quantum Gravity': {
        'icon': '🌌',
        'plots': [
            ('Spacetime Foam', fractal_dimension, None),
            ('Loop Quantum Gravity', topology_network, None),
            ('Holographic Principle', quantum_field_2d, 'inferno'),
            ('AdS/CFT Duality', manifold_3d, 'quantum'),
            ('String Theory Manifold', manifold_3d, 'dream'),
            ('Graviton Field', gauge_field, None),
            ('Black Hole Entropy', entropy_landscape, None),
            ('Hawking Radiation', wave_interference, None),
            ('Quantum Geometry', tensor_field, None),
            ('Spin Foam', topology_network, None),
            ('Causal Set', topology_network, None),
            ('Wheeler-DeWitt', quantum_tunneling, None),
            ('Planck Scale', bifurcation_diagram, None),
            ('Emergent Spacetime', eigenspace_projection, None),
            ('Gravitational Waves', attractor_dynamics, None),
            ('Cosmological Constant', manifold_3d, 'fire'),
        ]
    },
    
    'Stigmergy': {
        'icon': '🐜',
        'plots': [
            ('Pheromone Field', quantum_field_2d, 'turbo'),
            ('Swarm Intelligence', topology_network, None),
            ('Collective Behavior', attractor_dynamics, None),
            ('Emergence Landscape', entropy_landscape, None),
            ('Self-Organization', bifurcation_diagram, None),
            ('Coordination Network', topology_network, None),
            ('Trail Formation', gradient_flow_3d, None),
            ('Distributed Decision', phase_space_flow, None),
            ('Colony Dynamics', manifold_3d, 'plasma'),
            ('Information Flow', wave_interference, None),
            ('Adaptive Network', topology_network, None),
            ('Feedback Loops', correlation_matrix_3d, None),
            ('Quorum Sensing', quantum_field_2d, 'aurora'),
            ('Pattern Formation', fractal_dimension, None),
            ('Decentralized Control', tensor_field, None),
            ('Collective Memory', manifold_3d, 'quantum'),
        ]
    },
    
    'Neural Networks': {
        'icon': '🕸️',
        'plots': [
            ('Loss Landscape', manifold_3d, 'viridis'),
            ('Gradient Flow', gradient_flow_3d, None),
            ('Neural Tangent Kernel', quantum_field_2d, 'plasma'),
            ('Activation Manifold', eigenspace_projection, None),
            ('Weight Space Topology', topology_network, None),
            ('Optimization Dynamics', attractor_dynamics, None),
            ('Feature Space', correlation_matrix_3d, None),
            ('Attention Mechanism', tensor_field, None),
            ('Dropout Stochasticity', bifurcation_diagram, None),
            ('Batch Normalization', phase_space_flow, None),
            ('Residual Connections', wave_interference, None),
            ('Embedding Space', eigenspace_projection, None),
            ('Learned Representations', manifold_3d, 'dream'),
            ('Adversarial Landscape', entropy_landscape, None),
            ('Neural Architecture', topology_network, None),
            ('Generalization Surface', manifold_3d, 'turbo'),
        ]
    },
    
    'Chaos Theory': {
        'icon': '🌀',
        'plots': [
            ('Strange Attractor', attractor_dynamics, None),
            ('Bifurcation Cascade', bifurcation_diagram, None),
            ('Lyapunov Exponent', quantum_field_2d, 'fire'),
            ('Phase Space', phase_space_flow, None),
            ('Poincaré Section', wave_interference, None),
            ('Fractal Basin', fractal_dimension, None),
            ('Chaos Manifold', manifold_3d, 'turbo'),
            ('Sensitivity Field', entropy_landscape, None),
            ('Attractor Network', topology_network, None),
            ('Ergodic Flow', gradient_flow_3d, None),
            ('Mixing Dynamics', correlation_matrix_3d, None),
            ('Period Doubling', bifurcation_diagram, None),
            ('Nonlinear Resonance', wave_interference, None),
            ('Turbulent Flow', manifold_3d, 'plasma'),
            ('Control Landscape', quantum_field_2d, 'dream'),
            ('Synchronization', attractor_dynamics, None),
        ]
    },
    
    'String Theory': {
        'icon': '🎻',
        'plots': [
            ('Calabi-Yau Manifold', manifold_3d, 'quantum'),
            ('Compactification', topology_network, None),
            ('D-Brane Configuration', quantum_field_2d, 'aurora'),
            ('Moduli Space', eigenspace_projection, None),
            ('String Worldsheet', manifold_3d, 'dream'),
            ('Extra Dimensions', tensor_field, None),
            ('Mirror Symmetry', wave_interference, None),
            ('Supersymmetry', correlation_matrix_3d, None),
            ('M-Theory Landscape', entropy_landscape, None),
            ('Dualities Network', topology_network, None),
            ('Flux Compactification', gauge_field, None),
            ('Vacuum Manifold', manifold_3d, 'inferno'),
            ('Partition Function', quantum_field_2d, 'turbo'),
            ('Wilson Loops', attractor_dynamics, None),
            ('Heterotic String', bifurcation_diagram, None),
            ('Brane Dynamics', gradient_flow_3d, None),
        ]
    },
    
    'Information Theory': {
        'icon': '📡',
        'plots': [
            ('Shannon Entropy', entropy_landscape, None),
            ('Mutual Information', correlation_matrix_3d, None),
            ('Channel Capacity', quantum_field_2d, 'viridis'),
            ('Source Coding', bifurcation_diagram, None),
            ('Rate Distortion', manifold_3d, 'plasma'),
            ('Information Flow', phase_space_flow, None),
            ('Kolmogorov Complexity', fractal_dimension, None),
            ('Error Correction', topology_network, None),
            ('Fisher Information', tensor_field, None),
            ('Entropy Production', wave_interference, None),
            ('Data Compression', gradient_flow_3d, None),
            ('Algorithmic Information', entropy_landscape, None),
            ('Information Geometry', manifold_3d, 'quantum'),
            ('Coding Network', topology_network, None),
            ('Channel Noise', quantum_field_2d, 'ice'),
            ('Information Bottleneck', attractor_dynamics, None),
        ]
    },
    
    'Complex Systems': {
        'icon': '🔮',
        'plots': [
            ('Emergence Landscape', entropy_landscape, None),
            ('Network Dynamics', topology_network, None),
            ('Critical Phenomena', bifurcation_diagram, None),
            ('Self-Organization', manifold_3d, 'turbo'),
            ('Scale-Free Network', topology_network, None),
            ('Phase Transition', quantum_field_2d, 'fire'),
            ('Collective Behavior', attractor_dynamics, None),
            ('Complexity Measure', fractal_dimension, None),
            ('Adaptive System', gradient_flow_3d, None),
            ('Percolation Network', topology_network, None),
            ('Criticality Surface', manifold_3d, 'plasma'),
            ('Feedback Dynamics', phase_space_flow, None),
            ('Pattern Formation', wave_interference, None),
            ('Hierarchical Structure', eigenspace_projection, None),
            ('Resilience Landscape', entropy_landscape, None),
            ('Network Topology', correlation_matrix_3d, None),
        ]
    },
    
    'Manifold Learning': {
        'icon': '🗺️',
        'plots': [
            ('Riemannian Manifold', manifold_3d, 'dream'),
            ('Geodesic Flow', gradient_flow_3d, None),
            ('Tangent Space', eigenspace_projection, None),
            ('Curvature Tensor', tensor_field, None),
            ('Dimensionality Reduction', correlation_matrix_3d, None),
            ('Metric Tensor', quantum_field_2d, 'quantum'),
            ('Parallel Transport', gauge_field, None),
            ('Embedding Space', manifold_3d, 'aurora'),
            ('Local Coordinates', phase_space_flow, None),
            ('Differential Form', wave_interference, None),
            ('Manifold Atlas', topology_network, None),
            ('Connection Form', tensor_field, None),
            ('Isometric Embedding', eigenspace_projection, None),
            ('Laplacian Eigenmap', correlation_matrix_3d, None),
            ('Diffusion Map', quantum_field_2d, 'turbo'),
            ('UMAP Projection', manifold_3d, 'viridis'),
        ]
    },
    
    'Category Theory': {
        'icon': '📐',
        'plots': [
            ('Functor Mapping', topology_network, None),
            ('Natural Transformation', gradient_flow_3d, None),
            ('Categorical Diagram', topology_network, None),
            ('Adjunction Landscape', manifold_3d, 'quantum'),
            ('Topos Structure', eigenspace_projection, None),
            ('Monoidal Category', tensor_field, None),
            ('Yoneda Lemma', wave_interference, None),
            ('Categorical Limit', entropy_landscape, None),
            ('Sheaf Theory', quantum_field_2d, 'dream'),
            ('Higher Category', topology_network, None),
            ('Commutative Diagram', correlation_matrix_3d, None),
            ('Universal Property', manifold_3d, 'plasma'),
            ('Categorical Product', phase_space_flow, None),
            ('Equalizer Diagram', bifurcation_diagram, None),
            ('Enriched Category', manifold_3d, 'aurora'),
            ('Kan Extension', gradient_flow_3d, None),
        ]
    },
    
    'Quantum Computing': {
        'icon': '🔬',
        'plots': [
            ('Qubit State Space', quantum_field_2d, 'quantum'),
            ('Quantum Circuit', topology_network, None),
            ('Gate Fidelity', manifold_3d, 'ice'),
            ('Entanglement Network', topology_network, None),
            ('Quantum Algorithm', gradient_flow_3d, None),
            ('Error Correction Code', correlation_matrix_3d, None),
            ('Quantum Supremacy', entropy_landscape, None),
            ('Bloch Sphere', spherical_harmonics, None),
            ('Quantum Annealing', attractor_dynamics, None),
            ('QAOA Landscape', manifold_3d, 'viridis'),
            ('Variational Circuit', phase_space_flow, None),
            ('Quantum Channels', tensor_field, None),
            ('Measurement Basis', eigenspace_projection, None),
            ('Quantum Volume', quantum_field_2d, 'aurora'),
            ('Decoherence Rate', bifurcation_diagram, None),
            ('Quantum Advantage', manifold_3d, 'turbo'),
        ]
    },
    
    'Optimization Theory': {
        'icon': '📈',
        'plots': [
            ('Optimization Landscape', manifold_3d, 'viridis'),
            ('Gradient Descent', gradient_flow_3d, None),
            ('Convex Envelope', entropy_landscape, None),
            ('Lagrange Multiplier', quantum_field_2d, 'plasma'),
            ('Pareto Frontier', manifold_3d, 'dream'),
            ('Dual Problem', eigenspace_projection, None),
            ('Constraint Surface', tensor_field, None),
            ('KKT Conditions', phase_space_flow, None),
            ('Global Optimum', wave_interference, None),
            ('Convergence Dynamics', attractor_dynamics, None),
            ('Saddle Point', bifurcation_diagram, None),
            ('Barrier Method', quantum_field_2d, 'fire'),
            ('Trust Region', manifold_3d, 'quantum'),
            ('Line Search', gradient_flow_3d, None),
            ('Feasible Region', entropy_landscape, None),
            ('Objective Function', manifold_3d, 'turbo'),
        ]
    },
    
    'Statistical Mechanics': {
        'icon': '♨️',
        'plots': [
            ('Partition Function', entropy_landscape, None),
            ('Phase Diagram', quantum_field_2d, 'fire'),
            ('Free Energy', manifold_3d, 'inferno'),
            ('Correlation Function', correlation_matrix_3d, None),
            ('Ising Model', wave_interference, None),
            ('Critical Point', bifurcation_diagram, None),
            ('Boltzmann Distribution', gradient_flow_3d, None),
            ('Gibbs Ensemble', eigenspace_projection, None),
            ('Monte Carlo', attractor_dynamics, None),
            ('Spin Configuration', quantum_field_2d, 'quantum'),
            ('Thermodynamic Limit', manifold_3d, 'plasma'),
            ('Phase Transition', entropy_landscape, None),
            ('Order Parameter', phase_space_flow, None),
            ('Fluctuations', bifurcation_diagram, None),
            ('Entropy Landscape', entropy_landscape, None),
            ('Mean Field', tensor_field, None),
        ]
    },
    
    'Gauge Theory': {
        'icon': '🌊',
        'plots': [
            ('Yang-Mills Field', gauge_field, None),
            ('Gauge Transformation', gradient_flow_3d, None),
            ('Field Strength', quantum_field_2d, 'aurora'),
            ('Gauge Group', topology_network, None),
            ('Chern-Simons', manifold_3d, 'dream'),
            ('Instantons', wave_interference, None),
            ('Gauge Invariance', tensor_field, None),
            ('Connection Form', gauge_field, None),
            ('Curvature Form', quantum_field_2d, 'turbo'),
            ('Wilson Loop', attractor_dynamics, None),
            ('t Hooft Loop', topology_network, None),
            ('Monopole Field', manifold_3d, 'quantum'),
            ('Gauge Fixing', phase_space_flow, None),
            ('BRST Symmetry', eigenspace_projection, None),
            ('Lattice Gauge', correlation_matrix_3d, None),
            ('Gauge Anomaly', entropy_landscape, None),
        ]
    },
    
    'Differential Geometry': {
        'icon': '📏',
        'plots': [
            ('Gaussian Curvature', manifold_3d, 'quantum'),
            ('Geodesic Equation', gradient_flow_3d, None),
            ('Riemann Tensor', tensor_field, None),
            ('Christoffel Symbols', quantum_field_2d, 'dream'),
            ('Parallel Transport', gauge_field, None),
            ('Lie Derivative', phase_space_flow, None),
            ('Exterior Derivative', wave_interference, None),
            ('Metric Tensor', tensor_field, None),
            ('Killing Vector', gradient_flow_3d, None),
            ('Sectional Curvature', manifold_3d, 'plasma'),
            ('Ricci Flow', attractor_dynamics, None),
            ('Connection Form', gauge_field, None),
            ('Holonomy Group', topology_network, None),
            ('Gauss-Bonnet', entropy_landscape, None),
            ('Symplectic Form', quantum_field_2d, 'aurora'),
            ('Complex Manifold', eigenspace_projection, None),
        ]
    },
    
    'Network Science': {
        'icon': '🕸️',
        'plots': [
            ('Network Topology', topology_network, None),
            ('Centrality Measure', quantum_field_2d, 'viridis'),
            ('Community Structure', eigenspace_projection, None),
            ('Scale-Free', topology_network, None),
            ('Small World', topology_network, None),
            ('Degree Distribution', bifurcation_diagram, None),
            ('Clustering Coefficient', correlation_matrix_3d, None),
            ('Network Flow', phase_space_flow, None),
            ('Modularity', entropy_landscape, None),
            ('Percolation', quantum_field_2d, 'fire'),
            ('Network Motif', topology_network, None),
            ('PageRank', gradient_flow_3d, None),
            ('Epidemic Spread', wave_interference, None),
            ('Structural Balance', tensor_field, None),
            ('Link Prediction', manifold_3d, 'quantum'),
            ('Network Resilience', attractor_dynamics, None),
        ]
    },
    
    'Dynamical Systems': {
        'icon': '⚙️',
        'plots': [
            ('Phase Portrait', phase_space_flow, None),
            ('Poincaré Map', wave_interference, None),
            ('Limit Cycle', attractor_dynamics, None),
            ('Bifurcation', bifurcation_diagram, None),
            ('Stable Manifold', manifold_3d, 'turbo'),
            ('Basin of Attraction', quantum_field_2d, 'plasma'),
            ('Lyapunov Function', entropy_landscape, None),
            ('Hopf Bifurcation', gradient_flow_3d, None),
            ('Hamiltonian Flow', phase_space_flow, None),
            ('Period Doubling', bifurcation_diagram, None),
            ('Center Manifold', eigenspace_projection, None),
            ('Poincaré-Bendixson', wave_interference, None),
            ('Floquet Theory', correlation_matrix_3d, None),
            ('Normal Form', tensor_field, None),
            ('Invariant Torus', manifold_3d, 'quantum'),
            ('Ergodic Theory', attractor_dynamics, None),
        ]
    },
    
    'Quantum Field Theory': {
        'icon': '🌊',
        'plots': [
            ('Feynman Path', attractor_dynamics, None),
            ('Vacuum Energy', entropy_landscape, None),
            ('Propagator', quantum_field_2d, 'quantum'),
            ('Renormalization Flow', gradient_flow_3d, None),
            ('Field Configuration', manifold_3d, 'dream'),
            ('Scattering Amplitude', wave_interference, None),
            ('Loop Diagram', topology_network, None),
            ('Beta Function', bifurcation_diagram, None),
            ('Effective Action', entropy_landscape, None),
            ('Ward Identity', tensor_field, None),
            ('Anomaly Field', quantum_field_2d, 'aurora'),
            ('Instanton', manifold_3d, 'inferno'),
            ('Soliton Solution', quantum_tunneling, None),
            ('Correlation Function', correlation_matrix_3d, None),
            ('Conformal Field', manifold_3d, 'turbo'),
            ('Quantum Anomaly', phase_space_flow, None),
        ]
    }
}

# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: #00ffaa; font-size: 3em; margin-bottom: 0;'>✨ QUANTUM DREAM ATLAS ✨</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 1.2em;'>336 Nobel-Tier Visualizations Across 21 Advanced Domains</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #00ffaa; font-size: 0.9em; margin-bottom: 2em;'>🌌 Every plot dynamically generated • Random seed: {} • Pure aesthetic perfection 🌌</p>".format(st.session_state.master_seed), unsafe_allow_html=True)
    
    # Regenerate button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 REGENERATE ALL", use_container_width=True):
            st.session_state.master_seed = np.random.randint(0, 1000000)
            st.rerun()
    
    st.divider()
    
    # Generate all subjects
    plot_counter = 0
    for subject_name, subject_data in SUBJECTS.items():
        with st.expander(f"{subject_data['icon']} {subject_name.upper()}", expanded=False):
            st.markdown(f"<h3 style='color: #00ffaa;'>{subject_data['icon']} {subject_name}</h3>", unsafe_allow_html=True)
            
            plots = subject_data['plots']
            
            # Display in rows of 4 plots
            for row_start in range(0, len(plots), 4):
                row_plots = plots[row_start:row_start + 4]
                cols = st.columns(len(row_plots))
                
                for col_idx, (plot_title, plot_func, colorscale_arg) in enumerate(row_plots):
                    with cols[col_idx]:
                        try:
                            seed = get_seed(plot_counter)
                            subject_id = list(SUBJECTS.keys()).index(subject_name)
                            plot_id = row_start + col_idx
                            
                            if colorscale_arg:
                                fig = plot_func(seed, subject_id, plot_id, colorscale_arg)
                            else:
                                fig = plot_func(seed, subject_id, plot_id)
                            
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                            st.caption(f"**{plot_title}**")
                        except Exception as e:
                            st.error(f"Plot error: {str(e)[:50]}")
                        
                        plot_counter += 1
    
    # Footer
    st.divider()
    st.markdown("<p style='text-align: center; color: #666; font-size: 0.9em;'>Created with ❤️ for the beauty of science</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #444; font-size: 0.8em;'>Total plots: {plot_counter} | Master seed: {st.session_state.master_seed}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
