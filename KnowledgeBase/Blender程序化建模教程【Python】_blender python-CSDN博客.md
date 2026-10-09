---
title: "Blender程序化建模教程【Python】_blender python-CSDN博客"
source: "https://blog.csdn.net/shebao3333/article/details/122626171?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522d626a208f524224cc3b00e94868c47b5%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=d626a208f524224cc3b00e94868c47b5&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-122626171-null-null.142^v102^pc_search_result_base8&utm_term=blender%20python&spm=1018.2226.3001.4187"
author:
  - "[[成就一亿技术人!]]"
  - "[[hope_wisdom 发出的红包]]"
published:
created: 2025-11-07
description: "文章浏览阅读1.8w次，点赞33次，收藏146次。本文介绍如何使用Blender的Python API实现3D对象的自动化创建与动画设置，包括太阳系模型的生成。"
tags:
  - "clippings"
---
本文介绍如何使用Blender的Python API实现3D对象的自动化创建与动画设置，包括太阳系模型的生成。

你知道可以在 Blender 中通过编程来自动创建 3D 对象和动画吗？

Blender 是一款用于 3D 建模、雕刻、 [纹理](https://so.csdn.net/so/search?q=%E7%BA%B9%E7%90%86&spm=1001.2101.3001.7020) 、动画等方面的知名软件！随着版本的不断推出，这个工具已经慢慢在 CGI 行业中占据了一席之地，以至于现在有一些完全使用 Blender 制作的长篇电影，并且有像 Blender Guru 这样的 Youtube 频道完全专注于学习这个软件的来龙去…而且它是完全免费和开源的！

这在 3D 世界中引入了思维框架的深刻变化，因为它向人们展示了任何人都可以尝试并尝试这种艺术形式，并且可以取得令人难以置信的结果。

所以，是的——起初， [CGI](https://so.csdn.net/so/search?q=CGI&spm=1001.2101.3001.7020) 软件主要针对艺术家。但是 Blender 和它的竞争对手一样，也为开发人员提供了另一面：一种对 3D 场景进行编程的方法。

今天，我想展示 Blender 的这种编程方法如何让你立即创建一个像这样的基本太阳系：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/051a9f330ef7c1adf73631a663e3f321.png)

### 1、Blender的Python API

那么Blender 如何让我们“编程场景”？通过其 Python API 。

您可能想知道为什么拥有一个用于 3D 软件的 Python API 很有趣。我的意思是：为什么要 为编码人员嵌入一个创意工具？

事实上，在许多用例中， 自动化任务可能很有用：无论你是想快速随机化场景填充算法、计算对象并获取场景中的自定义统计信息，还是甚至从头开始创建一个可以用种子准确复制……有一种方法可以将 程序生成或定制工具 直接集成到 CG 上下文中，这真是一个了不起的机会！

此外，它是用 Python 编写的，这是一种以易于学习 而 闻名的语言 ，网上有无数的教程 ，这使得初学者可以很容易地深入（不用担心旧的 C/在我看来，基于 C++ 的 API 需要更高的编码技能……）。

顺便说一句：这个 Python API 不仅仅是对于想要成为开发者的闪亮玩具：它实际上 是 Blender 软件本身的一部分， 它 在程序内部 用作核心工具，尽管用户输入和结果都 包含在用户中-友好的用户界面 😉

API 本身，bpy（代表“Blender Python”），可以在 Blender 的特定文档中浏览；它被细分为几个子模块，其中 3 个最重要/最常用的是：

- bpy.context：它包含只读值的 getter 和 reader，这些值描述你当前的工作上下文，甚至是当前正在访问的区域（即窗口中的面板）
- bpy.data：它使你可以访问场景中的资源（对象、材质、网格……），以便你可以加载、添加或删除它们
- bpy.ops：这是 API 的真正核心——它允许你对对象或视图执行操作和调用运算符；它基本上是如何通过脚本模拟用户操作（例如选择对象、进入编辑模式、应用细分、更改为“平面”阴影、最大化窗口……）

今天，由于有了这个 API，我们可以专注于生成对象，因此我们将进行一些 过程生成。我已经在 其他 文章中讨论过这种方法的好处；粗略地说，程序生成是关于定义 一组规则 和 使用这些规则 自动创建有效实例的机器。最大的优势是，一旦你完成了引擎的制作，你可以创建任意数量的实例！此外，这一代可以非常快速、动态地取决于给定的条件和无限（例如对于像跑步者这样永无止境的游戏） ）。但是，当然，这通常比手工设计要难一些，因为需要教程序正确和错误的模式（即“规则”）。

程序生成是一个非常广泛和复杂的话题。生成引擎有很多工具——老实说，其中很大一部分依赖于 随机性。这个想法是从随机值开始，然后以某种方式“控制疯狂”，因此它是有效的。

### 2、生成随机太阳系并为其设置动画！

要了解 Blender 的 Python API 有多么强大以及如何使用它，让我们来看一个基本示例：实例化围绕太阳的一堆行星（都是简单的球体），大小、速度和颜色随机，并让行星围绕旋转太阳沿着圆形轨道。

现在，让我们明确一点：这 不是 物理仿真。我们将随机选择所有值，到太阳的距离、半径和行星的表面颜色之间没有任何逻辑关系。这里的目标只是尝试 Blender 的 Python API，看看如何进行一些程序生成并享受发光球体的乐趣😉

### 3、准备Blender

首先要做的事情：让我们看看如何为 Python 设置 Blender 并学习我们可以使用的不同工具。

注意：本教程使用 Blender 2.93 的屏幕截图。此外，我将使用 Blender 2.8+ 中的 Python API，因此请确保您的版本匹配🙂

当你想在 Blender 中使用 Python 时，非常有用的方法是在附加控制台的情况下 启动 Blender。如果你的 Python 脚本错误，这将允许您查看一些日志，这对于正确调试非常重要！

基本上，这个想法是从你的终端而不是从您的应用程序快捷方式启动 Blender。Blender 可执行文件的确切路径取决于你的操作系统；我使用的是 Mac，所以我所要做的就是打开一个终端，然后运行如下内容：

```
/Applications/Blender.app/Contents/MacOS/Blender &
1
```

Python 脚本的任何输出都会显示在这个控制台中🙂

现在，要准备好编码，只需转到 Blender 窗口顶部的“脚本” 选项卡 ：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/eef5b94bd01e9393a4e4589345a28b87.png)

这将在工作区中提供一组新面板，即：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/06d6a3f5786c6a47e710e50819249884.png)

1. 3D 视图：就像通常在 Blender 中的进入屏幕一样 - 你可以在 3D 中以“实心”阴影模式查看当前场景，并且可以像往常一样选择各种对象
2. Python 交互式控制台：定制的 Python 控制台对于浏览 Blender API 和探索 Python Blender 对象提供给我们的所有字段和方法非常有用；除了基本的 Python 内置插件，这个控制台还加载了 Blender 的内置插件和一些方便的变量，以便轻松访问 Blender API 的常用子模块
3. 信息面板：一 开始可能看起来很奇怪，但这个面板只是调试你所做的一切——这再次非常适合发现 API 的可能性！如果你尝试选择一个对象，或移动灯光，或习惯在 Blender 中执行的任何其他操作，你会看到它已记录在此面板中，因此可以了解应该访问 API 的哪个部分通过脚本重现此操作
4. 文本编辑器：你可能已经猜到了 ——这是我们今天的主菜；这是我们将创建、编辑、保存和运行Python 脚本的地方。它是一个简单的 Python 编辑器，带有语法高亮、行号，如果你打开侧边栏（在 Mac 上使用 <Ctrl + T> 或 <Cmd + T>），它甚至还有一些实用程序，如“查找和替换”。但是请注意，这个编辑器在功能方面是有限的，所以如果你打算做一件大事，最好使用一个好的外部 IDE，然后在准备好后在 Blender 中加载你的脚本
5. Outliner ：这是 你可能已经知道的另一个常见视图，它仅显示当前场景中对象的层次结构并允许快速访问
6. 属性面板：同样，此面板与 常见布局中的相同——你可以在其中查看和编辑场景的属性、渲染参数、世界设置以及当前所选对象的特定数据。  
	要检查一切是否正常，只需单击文本编辑器的“+ New”按钮，在 Python 脚本中编写基本的 print(‘hello world’) 行，然后单击右侧的“运行”图标（或使用快捷键 <Alt + P>；确保您的鼠标悬停在文本编辑器面板上以使其工作）。

如果一切设置正确，您应该会在连接的终端中看到“hello world”！  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/814e3b071a4d6ee645072267fcf1308e.png)

### 4、通过 API 创建基本网格

好吧！是时候在我们的场景中使用 Python 实例化一些对象了🙂

首先，删除所有初始对象：立方体、相机和灯光。我们不会使用这些。

现在，让我们编辑 Python 脚本来创建一个新的球体对象。这是使用 bpy.ops 子包完成的，更准确地说 是使用 mesh.primitive\_uv\_sphere\_add() 方法：

```
import bpy

bpy.ops.mesh.primitive_uv_sphere_add(
    radius=3,
    location=(0, 0, 0),
    scale=(1, 1, 1)
)
1234567
```

参数是不言自明的：我们在原点创建一个半径为 3 的球体，具有标准化的比例。（请随意浏览文档以获取有关可用选项的更多详细信息）

再次，通过单击“运行”图标或使用 <Alt + P> 短代码来运行您的代码。Nice！我们刚刚通过脚本在 3D 场景中创建了一个简单的 UV 球体！😉  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fcb836922bc1241f93a6ac66a57c3c51.png)

### 5、实例化我们的行星

我们现在能够生成一个球体——让我们看看生成多个球体是多么容易！

我们要做的是：

- 创建 N\_PLANETS 个对象，每个对象都是一个简单的 UV 球体
- 让每个对象在给定范围内使用随机半径
- 以及与原点（我们最终将放置太阳的地方）的随机距离取决于行星的索引（以便它们的轨道很好地分布）

制作 N\_PLANETS 实例很简单：我们只需将primitive\_uv\_sphere\_add() 调用包装在一个for循环中并多次运行它。为了保持脚本的可读性，我们实际上将这个实例化过程提取到一个 util 函数 create\_sphere() 中，并且我们将以随机半径和距离传递它。

要获得半径和距离的随机值，我们可以依赖 Python 内置的 random 模块。

此外，我们应该 正确命名我们的对象 ：这对于保持层次结构清晰是很有趣的，并且当我们稍后在脚本初始化时自动清理场景时，这将是必不可少的。对于 N\_PLANETS 球体，我将简单地将每个对象称为“Planet-00”、“Planet-01”等等。

这是更新的脚本：

```
from random import random
import bpy

def create_sphere(radius, distance_to_sun, obj_name):
    # instantiate a UV sphere with a given
    # radius, at a given distance from the
    # world origin point
    obj = bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=(distance_to_sun, 0, 0),
        scale=(1, 1, 1)
    )
    # rename the object
    bpy.context.object.name = obj_name
    # return the object reference
    return bpy.context.object

N_PLANETS = 6

for n in range(N_PLANETS):
    # get a random radius (a float in [1, 5])
    r = 1 + random() * 4
    # get a random distace to the origin point:
    # - an initial offset of 30 to get out of the sun's sphere
    # - a shift depending on the index of the planet
    # - a little "noise" with a random float
    d = 30 + n * 12 + (random() * 4 - 2)
    # instantiate the planet with these parameters
    # and a custom object name
    create_sphere(r, d, "Planet-{:02d}".format(n))
123456789101112131415161718192021222324252627282930
```

如果你再次运行它，应该会得到一组大小各异的排列良好的行星，它们都以正确的格式命名：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8246345db6ede8655760c27f1d436b8c.png)

### 6、添加太阳和半径环

以类似的方式，我们可以重用我们的 create\_sphere() 方法并制作另一个方法 create\_torus() ，为太阳和 一些圆环物体添加一个球体 以显示行星轨道：

```
from random import random
import bpy

def create_sphere(radius, distance_to_sun, obj_name):
    # ...

def create_torus(radius, obj_name):
    # (same as the create_sphere method)
    obj = bpy.ops.mesh.primitive_torus_add(
        location=(0, 0, 0),
        major_radius=radius,
        minor_radius=0.1,
        major_segments=60
    )
    bpy.context.object.name = obj_name
    return bpy.context.object

N_PLANETS = 6

for n in range(N_PLANETS):
    # ...
    # add the radius ring display
    create_torus(d, "Radius-{:02d}".format(n))

# add the sun sphere
sun = create_sphere(12, 0, "Sun")
1234567891011121314151617181920212223242526
```

太阳当然比行星大，并且环只是与行星球体同时放置，使用到原点的距离作为 torusradius 参数。

### 7、为对象设置材质和阴影

这很好，但那些物体有点沉闷，都是灰色和块状的。是时候研究两个重要的 3D 概念了： 对象的阴影和材质 ！

一般来说， 阴影 是 指对象如何对光做出反应并在 3D 视图或渲染中绘制。然而，在这里，我关注的是 “平滑”与“平坦”的 阴影，它决定 了物体的多面性。它基本上是第二层，进一步影响对象的渲染方式，但不会改变其实际几何形状：它只是看起来更平滑。

将对象设置为使用“平滑”着色真的很快：在 bpy.ops 中有一个专门的 shade\_smooth() 方法：

```
from random import random
import bpy

def create_sphere(radius, distance_to_sun, obj_name):
    # ...
    # apply smooth shading
    bpy.ops.object.shade_smooth()
    # return the object reference
    return bpy.context.object

def create_torus(radius, obj_name):
    # ...
    # apply smooth shading
    bpy.ops.object.shade_smooth()
    # return the object reference
    return bpy.context.object

# ...
123456789101112131415161718
```

这里真正的问题是：我们怎样才能给我们的行星一些 颜色 ，我们怎样才能让我们的太阳 发光？

答案是：感谢 材质和着色器！🙂

### 8、什么是材质和着色器？

这两个术语都可以用来谈论“定义对象如何渲染的属性调色板”。

粗略地说， 着色器 赋予 3D 对象颜色、光泽度、光泽度、粗糙度……想想看：在现实世界中，是什么让你说“嗯，这是木头”只是一整套线索：物体有点棕色，它会反射一点光，但不会像金属那样带有尖锐的光点，它不会像玻璃那样折射光线，也不会像镜子那样反射你的图像……嗯——随着不断改进的 CGI技术，所有这些属性现在都可以在我们的 3D 场景中建模和再现！

注意：当你真正的目标是逼真的渲染时，可能不得不深入研究 基于物理的渲染或 PBR。这就是 Blender 的 Cycles 引擎用于创建“感觉”真实的令人惊叹的逼真图片的原因，这主要归功于一堆复杂且经过精心调整的着色器。

在 Blender 中，着色器通常通过基于节点的图形编辑器（在“着色”选项卡中可用）进行编辑，它允许你链接和组合尽可能多的内置节点，以构建或多或少复杂的着色流。然而，在本教程中，我们将制作一个只有一个节点的超级简单着色器，因此我们将在 Python 脚本中完成所有操作🙂

然后材质 使用这些着色器并将它们应用到你的 3D 几何体。一个对象可能有多个材质槽，即它可以对几何体的不同部分使用不同的着色器，但我们今天不会深入讨论，我们将坚持每个对象使用一个材质槽。

### 9、选择着色器类型

好的，那么——我们应该创建什么着色器？

对于这个项目， 我正在使用 EEVEE 引擎 ，它也可以与着色器节点一起使用，即使它没有与 Cycles 引擎相同的节点类型。

但这很好，因为在这里，我想要的是两者中都存在的一个： Emission shader。你可以把它想象成一个有强度（“强度”参数）和颜色的大灯泡。它将使你的 3D 对象发光（因此你的对象将成为场景中的光源，我们将与其余网格进行交互！）并从本质上使其“发光”🙂

### 10、在Python 脚本中创建着色器

要完全通过脚本创建和分配着色器，我们必须：

- 使用 bpy.data 子模块
- 像使用着色器节点编辑器一样“编辑”这个着色器：我们将删除一些节点，添加其他节点，设置它们的属性并将它们链接在一起
- 检索对我们新创建的材料的引用
- 最后，将其添加到我们对象的材质数据（即插槽）中

让我们一步一步来。我们将从创建一个名为的新函数开始：

```
create_emission_shader()
1
```

它将接收一些强度和颜色参数，并将使用这些参数来设置具有发射节点和输出节点的基本 2 节点图。

思路是从基本节点模板入手，清除所有starter节点；然后，我们可以添加我们的发射和输出节点，通过更新其输入字段的值来配置发射节点，并在两个节点之间创建一个链接：

```
def create_emission_shader(color, strength, mat_name):
    # create a new material resource (with its
    # associated shader)
    mat = bpy.data.materials.new(mat_name)
    # enable the node-graph edition mode
    mat.use_nodes = True
    
    # clear all starter nodes
    nodes = mat.node_tree.nodes
    nodes.clear()

    # add the Emission node
    node_emission = nodes.new(type="ShaderNodeEmission")
    # (input[0] is the color)
    node_emission.inputs[0].default_value = color
    # (input[1] is the strength)
    node_emission.inputs[1].default_value = strength
    
    # add the Output node
    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    
    # link the two nodes
    links = mat.node_tree.links
    link = links.new(node_emission.outputs[0], node_output.inputs[0])

    # return the material reference
    return mat
123456789101112131415161718192021222324252627
```

现在很容易使用这种方法来创建我们的材质资源并将它们应用于我们的对象。我们需要一种白色发光材料用于环，一种黄色发光材料用于太阳，每个行星需要一种颜色随机的发光材料（尽管我会添加更多蓝色以获得更好的整体色彩平衡😉）：

```
#...

N_PLANETS = 6

ring_mat = create_emission_shader(
    (1, 1, 1, 1), 1, "RingMat"
)
for n in range(N_PLANETS):
    # ...
    planet = create_sphere(r, d, "Planet-{:02d}".format(n))
    planet.data.materials.append(
        create_emission_shader(
            (random(), random(), 1, 1),
            2,
            "PlanetMat-{:02d}".format(n)
        )
    )
    # add the radius ring display
    ring = create_torus(d, "Radius-{:02d}".format(n))
    ring.data.materials.append(ring_mat)

# add the sun sphere
sun = create_sphere(12, 0, "Sun")
sun.data.materials.append(
    create_emission_shader(
        (1, 0.66, 0.08, 1), 10, "SunMat"
    )
)
12345678910111213141516171819202122232425262728
```

如果你将 3D 视图中的着色模式更改为“渲染”，删除当前场景中的所有对象并再次运行脚本，会看到它们现在拥有漂亮的发光材质！  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a28264fca14df87e107bc842339fd602.png)

请注意，运行脚本后，你甚至可以转到“着色”选项卡，选择带有着色器的对象并查看着色器图：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3db28c064847cc322dd06ea164460209.png)

它看起来符合预期：两个节点，一个发射和一个输出，两者之间的链接，以及“强度”和“颜色”属性的一些自定义值。

注意：当你最初打开面板时，节点将在彼此顶部的中间全部打包在一起。在这里，我为演示手动移动了节点，但实际上也可以在代码中使用.location 属性来执行此操作。

### 11、为行星设置动画

我们即将拥有最终的“太阳系发电机”！需要注意的最后一件事是为行星设置动画，以便它们随着时间的推移围绕太阳旋转。

为此，我们将使用 Blender 的 动画曲线 系统（F-curves）。它是关键帧动画的更 高级 版本 ，你可以在其中 指定对象的一个或多个属性随时间变化的关键值 （例如，在给定帧强制其位置、旋转、缩放……）以及它们之间的 插值 。

这是具有三种可能插值的同一组关键点的示例（从左到右：常数、线性和贝塞尔曲线）：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ddbc3cb9b4fa677a87b9d480da11bfdd.png)

你会看到插值如何影响中间自动计算值（我们手动定义的关键帧之间的所有片段）以及它如何影响对象此属性的整体演变。假设这条线代表一架小直升机的高度：

- 在不断插值的情况下，直升机只是不停地上下俯冲，然后带着一点高原直线前进
- 使用线性插值，直升机以甜美的对角线从一个点滑到另一个点，但是当它到达极值时它仍然会突然改变方向
- 通过贝塞尔插值，直升机在到达那些山谷和山脊之前减速并平稳地更新其航线

“最佳插值”取决于你想要的动画类型。在我们的例子中，我们应该选择线性插值，以便行星沿着它们的轨道以规则的速度移动。

要为我们的行星创建一些动画，只需要使用animation\_data\_create() 方法和我们对象的animation\_data 字段来创建然后编辑Z-rotation 属性的F 曲线。这个属性实际上是对象的 3D 向量属性的一部分，称为“欧拉旋转”（更多关于 Blender 旋转模式 here），其中 Z 轴是第三个分量，即索引 2 处的那个（因为分量是 0 -索引）。

一旦我们抓住了这个属性，将简单地添加两个关键帧：一个用于 起始帧 （我们当前的旋转为 0），另一个用于 结束帧 （ 一个或多个围绕太阳的半圆随机旋转） ，因此行星具有 不同的速度）。我们将确保这些关键帧使用线性插值模式😉

注意：所有旋转都必须使用 弧度写入。

```
from math import pi

# ...

N_PLANETS = 6
START_FRAME = 1
END_FRAME = 200

for n in range(N_PLANETS):
    # ...

    # setup the planet animation data
    planet.animation_data_create()
    planet.animation_data.action = bpy.data.actions.new(name="RotationAction")
    fcurve = planet.animation_data.action.fcurves.new(
        data_path="rotation_euler", index=2
    )
    k1 = fcurve.keyframe_points.insert(
        frame=START_FRAME,
        value=0
    )
    k1.interpolation = "LINEAR"
    k2 = fcurve.keyframe_points.insert(
        frame=END_FRAME,
        value=(2 + random() * 2) * pi
    )
    k2.interpolation = "LINEAR"
123456789101112131415161718192021222324252627
```

如果你清理场景并重新运行脚本，就会看到……什么都没有发生！即使你播放动画，行星也不会移动！

那是因为，事实上，它们是在旋转……但不是围绕正确的支点！目前，这颗行星只是在旋转，它们围绕着它们的局部 Z 轴旋转。

要解决这个问题，我们只需要更改对象的轴心点并将其捕捉回位于世界原点（与太阳相同的位置）的光标：

```
# ...

for n in range(N_PLANETS):
    # ...

    # set planet as active object
    bpy.context.view_layer.objects.active = planet
    planet.select_set(True)
    # set object origin at world origin
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")
    # setup the planet animation data
    # ...
123456789101112
```

现在，如果你将鼠标悬停在 3D 视图上并按下空格键，时间将开始流动，动画将开始播放，让我们的行星围绕太阳旋转！🙂

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ad5a43cb41dfc36fe2b5299d9da105ed.png)

### 12、在脚本初始化时自动清理场景

运行脚本后，我们在场景中创建了两种类型的 资源 ：行星、太阳和轨道环的 3D 对象；以及行星的动态材料。

摆脱它非常简单：我们只需要使用 bpy.data 子模块来检查我们的对象和材料，检查名称并删除我们在运行脚本时创建的名称：

```
# ...

def delete_object(name):
    # try to find the object by name
    if name in bpy.data.objects:
        # if it exists, select it and delete it
        obj = bpy.data.objects[name]
        obj.select_set(True)
        bpy.ops.object.delete(use_global=False)

N_PLANETS = 6
START_FRAME = 1
END_FRAME = 200

# clean scene + planet materials
delete_object("Sun")
for n in range(N_PLANETS):
    delete_object("Planet-{:02d}".format(n))
    delete_object("Radius-{:02d}".format(n))
for m in bpy.data.materials:
    bpy.data.materials.remove(m)

# ...
1234567891011121314151617181920212223
```

现在，你可以根据需要多次运行脚本：每次首先 清理场景， 以便只获得我们的“太阳能系统”的一个实例🙂

### 13、自动设置场景属性、渲染引擎和 3D 视图

为了获得更好的视觉效果，带有一点点 绽放效果、深色背景和没有网格或 X/Y 轴，甚至可以在脚本的开头添加以下代码片段来设置具有良好设置的场景：

```
# ...

def find_3dview_space():
    # find the 3D view panel and its screen space
    area = None
    for a in bpy.data.window_managers[0].windows[0].screen.areas:
        if a.type == "VIEW_3D":
            area = a
            break
    return area.spaces[0] if area else bpy.context.space_data

def setup_scene():
    # (set a black background)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    # (make sure we use the EEVEE render engine + enable bloom effect)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_bloom = True
    # (set the animation start/end/current frames)
    scene.frame_start = START_FRAME
    scene.frame_end = END_FRAME
    scene.frame_current = START_FRAME
    # get the current 3D view (among all visible windows
    # in the workspace)
    space = find_3dview_space()
    # apply a "rendered" shading mode + hide all
    # additional markers, grids, cursors...
    space.shading.type = 'RENDERED'
    space.overlay.show_floor = False
    space.overlay.show_axis_x = False
    space.overlay.show_axis_y = False
    space.overlay.show_cursor = False
    space.overlay.show_object_origins = False

N_PLANETS = 6
START_FRAME = 1
END_FRAME = 200

# setup scene settings
setup_scene()

# ...

# deselect all objects
bpy.ops.object.select_all(action='DESELECT')
123456789101112131415161718192021222324252627282930313233343536373839404142434445
```

### 14、结束语

Blender 不是唯一允许你对场景进行编程和自动执行任务的 3D 软件；但它不负众望，随着每个新版本的推出，Blender 逐渐成为一个 可靠的一体化 CG 制作解决方案，从使用油性铅笔的故事板到基于节点的合成。

你可以使用 Python 脚本和几个额外的包来批量生成对象实例化程序，设置渲染设置甚至获取当前项目的自定义统计信息，这一点非常棒！对我来说，这是一种减轻繁琐任务负担的方式，同时也让开发人员参加聚会，将这个创意工具社区扩展到艺术家之外。

在这篇文章中，我们已经看到，用不到一百行 Python 代码，我们就可以创建一个具有动态和随机网格、材质和动画的基本程序太阳系！

---

原文链接： [Blender程序化建模入门 — BimAnt](http://www.bimant.com/blog/blender-procedural-content-generation/)

实付 元

[使用余额支付](https://blog.csdn.net/shebao3333/article/details/)

点击重新获取

扫码支付

钱包余额 0

抵扣说明：

1.余额是钱包充值的虚拟货币，按照1:1的比例进行支付金额的抵扣。  
2.余额无法直接购买下载，可以购买VIP、付费专栏及课程。

[余额充值](https://i.csdn.net/#/wallet/balance/recharge)

举报

[![](https://i-operation.csdnimg.cn/images/582f17d21e394143b95ee3394233ae12.gif)](https://mall.csdn.net/vip?utm_source=25618_vip_blogrighticon) [![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/Group.png) 点击体验  
DeepSeekR1满血版](https://ai.csdn.net/?utm_source=cknow_pc_blogdetail&spm=1001.2101.3001.10583) 隐藏侧栏 搜索 ![程序员都在用的中文IT技术交流社区](https://g.csdnimg.cn/side-toolbar/3.6/images/qr_app.png)

程序员都在用的中文IT技术交流社区

![专业的中文 IT 技术社区，与千万技术人共成长](https://g.csdnimg.cn/side-toolbar/3.6/images/qr_wechat.png)

专业的中文 IT 技术社区，与千万技术人共成长

![关注【CSDN】视频号，行业资讯、技术分享精彩不断，直播好礼送不停！](https://g.csdnimg.cn/side-toolbar/3.6/images/qr_video.png)

关注【CSDN】视频号，行业资讯、技术分享精彩不断，直播好礼送不停！

客服 返回顶部

![](https://i-blog.csdnimg.cn/blog_migrate/051a9f330ef7c1adf73631a663e3f321.png) ![](https://i-blog.csdnimg.cn/blog_migrate/eef5b94bd01e9393a4e4589345a28b87.png) ![](https://i-blog.csdnimg.cn/blog_migrate/06d6a3f5786c6a47e710e50819249884.png) ![](https://i-blog.csdnimg.cn/blog_migrate/814e3b071a4d6ee645072267fcf1308e.png) ![](https://i-blog.csdnimg.cn/blog_migrate/fcb836922bc1241f93a6ac66a57c3c51.png) ![](https://i-blog.csdnimg.cn/blog_migrate/8246345db6ede8655760c27f1d436b8c.png) ![](https://i-blog.csdnimg.cn/blog_migrate/a28264fca14df87e107bc842339fd602.png) ![](https://i-blog.csdnimg.cn/blog_migrate/3db28c064847cc322dd06ea164460209.png) ![](https://i-blog.csdnimg.cn/blog_migrate/ddbc3cb9b4fa677a87b9d480da11bfdd.png) ![](https://i-blog.csdnimg.cn/blog_migrate/ad5a43cb41dfc36fe2b5299d9da105ed.png)