// JavaScript for neuron activity visualization will go here.
console.log("script.js loaded");

document.addEventListener('DOMContentLoaded', () => {
    const neuronCanvas = document.getElementById('neuronCanvas');
    const ctx = neuronCanvas.getContext('2d');
    const colorbarCanvas = document.getElementById('colorbarCanvas');
    const colorbarCtx = colorbarCanvas.getContext('2d');

    const playPauseBtn = document.getElementById('playPauseBtn');
    const speedControl = document.getElementById('speedControl');
    const speedValueSpan = document.getElementById('speedValue');
    const frameSlider = document.getElementById('frameSlider');
    const currentFrameSpan = document.getElementById('currentFrame');
    const behaviorLabelSpan = document.getElementById('behaviorLabel');
    const minActivityLabel = document.getElementById('minActivityLabel');
    const maxActivityLabel = document.getElementById('maxActivityLabel');

    // 新增：获取激活阈值滑动条和值显示的元素
    const activationThresholdSlider = document.getElementById('activationThresholdSlider');
    const activationThresholdValueSpan = document.getElementById('activationThresholdValue');

    let neuronData = null;
    let currentFrame = 0;
    let isPlaying = false;
    let animationSpeed = parseInt(speedControl.value, 10); // FPS
    let lastFrameTime = 0;
    let animationFrameId = null;

    const canvasWidth = 600; // 主Canvas宽度
    const canvasHeight = 500; // 主Canvas高度
    neuronCanvas.width = canvasWidth;
    neuronCanvas.height = canvasHeight;
    
    const neuronRadius = 7; // 神经元绘制半径
    const padding = 40; // Canvas边缘的内边距

    // 修改：将 ACTIVATION_THRESHOLD 设为变量，并从滑动条初始化
    let ACTIVATION_THRESHOLD = parseFloat(activationThresholdSlider.value); 
    // const MAX_CONNECTION_DISTANCE = 100; // 激活神经元之间绘制连接线的最大距离 (像素) - 移除距离限制
    const CONNECTION_LINE_WIDTH = 1.5; // 连接线的宽度
    const MIN_CONNECTION_OPACITY = 0.2; // 连接线最小透明度
    const MAX_CONNECTION_OPACITY = 0.8; // 连接线最大透明度

    // 颜色映射表 (近似Viridis: 紫-蓝-青-绿-黄)
    const viridisColorMap = [
        { stop: 0.0, color: { r: 68, g: 1, b: 84 } },    // #440154
        { stop: 0.25, color: { r: 59, g: 82, b: 139 } }, // #3B528B
        { stop: 0.5, color: { r: 33, g: 144, b: 140 } },// #21908C
        { stop: 0.75, color: { r: 93, g: 201, b: 99 } }, // #5DC963
        { stop: 1.0, color: { r: 253, g: 231, b: 37 } }  // #FDE725
    ];

    function interpolateColor(value, colorMap) {
        if (value <= colorMap[0].stop) return colorMap[0].color;
        if (value >= colorMap[colorMap.length - 1].stop) return colorMap[colorMap.length - 1].color;

        for (let i = 0; i < colorMap.length - 1; i++) {
            const s1 = colorMap[i].stop;
            const c1 = colorMap[i].color;
            const s2 = colorMap[i + 1].stop;
            const c2 = colorMap[i + 1].color;

            if (value >= s1 && value <= s2) {
                const t = (value - s1) / (s2 - s1);
                const r = Math.round(c1.r + (c2.r - c1.r) * t);
                const g = Math.round(c1.g + (c2.g - c1.g) * t);
                const b = Math.round(c1.b + (c2.b - c1.b) * t);
                return { r, g, b };
            }
        }
        return colorMap[0].color; // Fallback
    }

    function getColorForActivity(value) {
        const color = interpolateColor(value, viridisColorMap);
        return `rgb(${color.r},${color.g},${color.b})`;
    }

    function mapToCanvasCoords(neuronX, neuronY) {
        // 原始坐标 neuronX, neuronY 假设在 0.0 到 1.0 的范围内。
        // 我们希望将这个 [0,1]x[0,1] 的区域映射到画布内边距定义的绘图区域。
        const drawableWidth = canvasWidth - 2 * padding;
        const drawableHeight = canvasHeight - 2 * padding;

        const canvasX = padding + neuronX * drawableWidth;
        const canvasY = padding + neuronY * drawableHeight;
        
        return { x: canvasX, y: canvasY };
    }

    function drawNeurons(frameIdx) {
        if (!neuronData) return;

        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        const activitiesForFrame = neuronData.activity_data[frameIdx];
        const activatedNeurons = []; // 存储当前帧激活的神经元信息

        // console.log(`Frame ${frameIdx}: Processing ${neuronData.neuron_positions.length} neurons.`);

        // 第一遍：绘制神经元并收集激活的神经元
        neuronData.neuron_positions.forEach((neuronPos, index) => {
            const activity = activitiesForFrame[index];
            const color = getColorForActivity(activity);
            const { x, y } = mapToCanvasCoords(neuronPos.x, neuronPos.y);

            ctx.beginPath();
            ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 0.5;
            ctx.stroke();

            if (activity >= ACTIVATION_THRESHOLD) {
                activatedNeurons.push({ x, y, activity });
            }
        });

        // 调试信息：显示激活神经元的数量和活动值
        if (activatedNeurons.length > 0) {
            console.log(`Frame ${frameIdx}: Found ${activatedNeurons.length} activated neurons. Activities:`, activatedNeurons.map(n => n.activity.toFixed(2)));
        }

        // 第二遍：绘制激活神经元之间的连接
        if (activatedNeurons.length < 2) {
            // console.log(`Frame ${frameIdx}: Less than 2 activated neurons, skipping connections.`);
            return; // 如果激活的神经元少于2个，则无法形成连接
        }

        ctx.lineWidth = CONNECTION_LINE_WIDTH;
        for (let i = 0; i < activatedNeurons.length; i++) {
            for (let j = i + 1; j < activatedNeurons.length; j++) {
                const n1 = activatedNeurons[i];
                const n2 = activatedNeurons[j];

                // 根据平均活动强度调整透明度 (更活跃的连接更不透明)
                const avgActivity = (n1.activity + n2.activity) / 2;
                // 将活动值 (假设在ACTIVATION_THRESHOLD到1之间) 映射到透明度范围
                const opacityRange = MAX_CONNECTION_OPACITY - MIN_CONNECTION_OPACITY;
                const activityRange = 1.0 - ACTIVATION_THRESHOLD;
                let opacity = MIN_CONNECTION_OPACITY;
                if (activityRange > 0) { // 避免除以零
                    opacity = MIN_CONNECTION_OPACITY + ((avgActivity - ACTIVATION_THRESHOLD) / activityRange) * opacityRange;
                } else if (avgActivity >= ACTIVATION_THRESHOLD) { // 如果所有激活的都在阈值上
                    opacity = MAX_CONNECTION_OPACITY;
                }
                opacity = Math.max(MIN_CONNECTION_OPACITY, Math.min(MAX_CONNECTION_OPACITY, opacity));

                // 使用激活神经元的平均颜色，并调整透明度
                const color1 = interpolateColor(n1.activity, viridisColorMap);
                const color2 = interpolateColor(n2.activity, viridisColorMap);
                const r_avg = Math.round((color1.r + color2.r) / 2);
                const g_avg = Math.round((color1.g + color2.g) / 2);
                const b_avg = Math.round((color1.b + color2.b) / 2);
                ctx.strokeStyle = `rgba(${r_avg}, ${g_avg}, ${b_avg}, ${opacity})`;
                
                // 添加缺失的路径定义命令
                ctx.beginPath(); 
                ctx.moveTo(n1.x, n1.y);
                ctx.lineTo(n2.x, n2.y);
                ctx.stroke();
            }
        }
    }

    function drawColorbar() {
        if (!neuronData) return;
        const w = colorbarCanvas.width;
        const h = colorbarCanvas.height;
        colorbarCtx.clearRect(0, 0, w, h);

        const gradient = colorbarCtx.createLinearGradient(0, h, 0, 0); // 从下往上绘制，匹配活动值从低到高

        viridisColorMap.forEach(item => {
            gradient.addColorStop(item.stop, `rgb(${item.color.r},${item.color.g},${item.color.b})`);
        });

        colorbarCtx.fillStyle = gradient;
        colorbarCtx.fillRect(0, 0, w, h);

        minActivityLabel.textContent = `Min: ${neuronData.global_min_activity.toFixed(2)}`;
        maxActivityLabel.textContent = `Max: ${neuronData.global_max_activity.toFixed(2)}`;
    }
    
    function updateUI(frameIdx) {
        if (!neuronData) return;
        currentFrame = frameIdx;
        frameSlider.value = frameIdx;
        currentFrameSpan.textContent = `Frame: ${frameIdx + 1} / ${neuronData.num_frames}`;
        behaviorLabelSpan.textContent = `Behavior: ${neuronData.behavior_labels[frameIdx]}`;
    }

    function animate(timestamp) {
        if (!isPlaying) {
            animationFrameId = null; // Clear ID if stopped
            return;
        }

        const deltaTime = timestamp - lastFrameTime;
        const interval = 1000 / animationSpeed;

        if (deltaTime >= interval) {
            lastFrameTime = timestamp - (deltaTime % interval);
            currentFrame++;
            if (currentFrame >= neuronData.num_frames) {
                currentFrame = 0; // Loop animation
            }
            drawNeurons(currentFrame);
            updateUI(currentFrame);
        }
        animationFrameId = requestAnimationFrame(animate);
    }

    function playPause() {
        isPlaying = !isPlaying;
        playPauseBtn.textContent = isPlaying ? 'Pause' : 'Play';
        if (isPlaying) {
            lastFrameTime = performance.now();
            if (!animationFrameId) { // 避免重复启动
                animationFrameId = requestAnimationFrame(animate);
            }
        } else {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
        }
    }

    function updateToFrame(frameIdx) {
        if (!neuronData || frameIdx < 0 || frameIdx >= neuronData.num_frames) return;
        
        const wasPlaying = isPlaying;
        if (wasPlaying) { // 拖动滑块时暂停播放
            playPause(); 
        }
        
        currentFrame = frameIdx;
        drawNeurons(currentFrame);
        updateUI(currentFrame);

        if (wasPlaying && !isPlaying) { // 如果之前是播放状态，并且因为拖动滑块暂停了，则恢复播放
             // playPause(); // 可选：拖动后是否自动恢复播放
        }
    }
    
    async function init() {
        try {
            const response = await fetch('/api/neuron_data');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            neuronData = await response.json();

            if (neuronData.error) {
                console.error("Error fetching data:", neuronData.error);
                alert("Error fetching neuron data: " + neuronData.error);
                return;
            }
            
            console.log("Neuron data loaded:", neuronData);

            frameSlider.max = neuronData.num_frames - 1;
            frameSlider.value = 0;
            
            drawColorbar();
            updateToFrame(0); // Draw initial frame

        } catch (error) {
            console.error('Failed to initialize visualization:', error);
            alert('Failed to load visualization data. Please check the console.');
        }

        playPauseBtn.addEventListener('click', playPause);

        speedControl.addEventListener('input', (e) => {
            animationSpeed = parseInt(e.target.value, 10);
            speedValueSpan.textContent = `${animationSpeed} FPS`;
        });

        frameSlider.addEventListener('input', (e) => {
            const newFrame = parseInt(e.target.value, 10);
            updateToFrame(newFrame);
        });

        // 新增：激活阈值滑动条的事件监听器
        activationThresholdSlider.addEventListener('input', (e) => {
            ACTIVATION_THRESHOLD = parseFloat(e.target.value);
            activationThresholdValueSpan.textContent = ACTIVATION_THRESHOLD.toFixed(2);
            // 阈值改变后，需要重绘当前帧以反映连接变化
            if (neuronData) { // 确保数据已加载
                drawNeurons(currentFrame);
                // updateUI(currentFrame); // updateUI 主要更新帧号和行为标签，这里不需要除非有其他依赖
            }
        });
    }

    init();
}); 