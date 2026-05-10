"""
创建示例学生手册PDF

用于演示和测试RAG系统。
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


def create_sample_handbook(output_path: str = "data/pdfs/sample_student_handbook.pdf"):
    """创建示例学生手册PDF"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 注册中文字体
    try:
        pdfmetrics.registerFont(TTFont('SimSun', 'simsun.ttc'))
        pdfmetrics.registerFont(TTFont('SimHei', 'simhei.ttf'))
        chinese_font = 'SimSun'
        chinese_font_bold = 'SimHei'
    except:
        # 如果找不到中文字体，使用默认字体
        chinese_font = 'Helvetica'
        chinese_font_bold = 'Helvetica-Bold'

    # 创建PDF文档
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    # 定义样式
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'ChineseTitle',
        parent=styles['Heading1'],
        fontName=chinese_font_bold,
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=30
    )

    chapter_style = ParagraphStyle(
        'ChineseChapter',
        parent=styles['Heading1'],
        fontName=chinese_font_bold,
        fontSize=18,
        spaceAfter=20,
        spaceBefore=20
    )

    section_style = ParagraphStyle(
        'ChineseSection',
        parent=styles['Heading2'],
        fontName=chinese_font_bold,
        fontSize=14,
        spaceAfter=12,
        spaceBefore=12
    )

    body_style = ParagraphStyle(
        'ChineseBody',
        parent=styles['BodyText'],
        fontName=chinese_font,
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        leading=18
    )

    # 构建内容
    story = []

    # 封面
    story.append(Spacer(1, 5*cm))
    story.append(Paragraph("XX大学学生手册", title_style))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("（2024年版）", body_style))
    story.append(PageBreak())

    # 第一章：奖学金评定
    story.append(Paragraph("第一章 奖学金评定办法", chapter_style))

    story.append(Paragraph("第一条 评定目的", section_style))
    story.append(Paragraph(
        "为激励学生勤奋学习、全面发展，根据《普通高等学校学生管理规定》，结合我校实际，制定本办法。",
        body_style
    ))

    story.append(Paragraph("第二条 奖学金类别", section_style))
    story.append(Paragraph(
        "我校设立以下奖学金类别：国家奖学金、国家励志奖学金、校级一等奖学金、校级二等奖学金、校级三等奖学金，以及各类专项奖学金。",
        body_style
    ))

    story.append(Paragraph("第三条 评定条件", section_style))
    story.append(Paragraph(
        "申请奖学金的学生应当具备以下基本条件：热爱社会主义祖国，拥护中国共产党的领导；遵守宪法和法律，遵守学校规章制度；诚实守信，道德品质优良；在校期间学习成绩优异，社会实践、创新能力、综合素质等方面表现突出。",
        body_style
    ))

    story.append(Paragraph("第四条 成绩要求", section_style))
    story.append(Paragraph(
        "申请国家奖学金者，学习成绩排名与综合考评成绩排名均位于前10%；申请校级一等奖学金者，学习成绩排名位于前15%；申请校级二等奖学金者，学习成绩排名位于前25%；申请校级三等奖学金者，学习成绩排名位于前40%。",
        body_style
    ))

    story.append(Paragraph("第五条 评定程序", section_style))
    story.append(Paragraph(
        "奖学金评定按以下程序进行：学生本人申请，填写《奖学金申请表》；班级评议小组评议；学院审核并公示；学校审批并公示；发放奖学金及证书。",
        body_style
    ))

    story.append(PageBreak())

    # 第二章：缓考申请
    story.append(Paragraph("第二章 缓考申请规定", chapter_style))

    story.append(Paragraph("第六条 申请条件", section_style))
    story.append(Paragraph(
        "学生因以下原因不能参加课程考核的，可以申请缓考：因病不能参加考试（需提供二级甲等以上医院证明）；因家庭重大变故需要离校处理；因参加学校组织的重大活动或竞赛；其他经学校认定的正当理由。",
        body_style
    ))

    story.append(Paragraph("第七条 申请时间", section_style))
    story.append(Paragraph(
        "缓考申请应当在课程考核前提出。因病等突发情况不能提前申请的，应当在考核后3个工作日内补办申请手续，并提供相关证明材料。",
        body_style
    ))

    story.append(Paragraph("第八条 申请流程", section_style))
    story.append(Paragraph(
        "缓考申请流程如下：学生登录教务系统填写《缓考申请表》；下载打印申请表，附相关证明材料；辅导员审核签字；学院教学秘书审核；教务处审批；审批通过后，学生参加下学期开学初组织的缓考考试。",
        body_style
    ))

    story.append(Paragraph("第九条 成绩记载", section_style))
    story.append(Paragraph(
        "缓考课程的成绩按实际考试成绩记载。缓考不及格者，不再安排补考，应当重修该课程。",
        body_style
    ))

    story.append(PageBreak())

    # 第三章：学籍管理
    story.append(Paragraph("第三章 学籍管理规定", chapter_style))

    story.append(Paragraph("第十条 入学与注册", section_style))
    story.append(Paragraph(
        "新生应当持录取通知书和学校规定的有关证件，按期到校办理入学手续。因故不能按期入学者，应当向学校请假，假期一般不得超过两周。未请假或请假逾期者，除因不可抗力等正当事由外，视为放弃入学资格。",
        body_style
    ))

    story.append(Paragraph("第十一条 转专业", section_style))
    story.append(Paragraph(
        "学生入学后，原则上应当在录取专业完成学业。符合下列条件之一者，可以申请转专业：确有特长，转专业更能发挥其特长；因疾病或生理缺陷，经学校指定医院证明不宜在原专业学习；确有特殊困难，不转专业无法继续学习；学校根据社会对人才需求情况的发展变化，经学生同意，必要时可以适当调整学生所学专业。",
        body_style
    ))

    story.append(Paragraph("第十二条 休学与复学", section_style))
    story.append(Paragraph(
        "学生可以分阶段完成学业。学生在校最长年限（含休学）为所学专业学制加两年。学生休学一般以一年为期，累计不得超过两年。学生休学期满，应当于学期开学前向学校提出复学申请，经学校复查合格，方可复学。",
        body_style
    ))

    story.append(PageBreak())

    # 第四章：违纪处分
    story.append(Paragraph("第四章 违纪处分规定", chapter_style))

    story.append(Paragraph("第十三条 处分种类", section_style))
    story.append(Paragraph(
        "对有违法、违规、违纪行为的学生，学校给予批评教育或纪律处分。纪律处分的种类分为：警告；严重警告；记过；留校察看；开除学籍。",
        body_style
    ))

    story.append(Paragraph("第十四条 考试违纪", section_style))
    story.append(Paragraph(
        "学生在考试中有下列行为之一的，应当认定为考试违纪：携带规定以外的物品进入考场或未放在指定位置；未在规定的座位参加考试；考试开始信号发出前答题或考试结束信号发出后继续答题；在考试过程中旁窥、交头接耳、互打暗号或手势；在考场或教育考试机构禁止的范围内，喧哗、吸烟或实施其他影响考场秩序的行为；未经考试工作人员同意在考试过程中擅自离开考场；将试卷、答卷（含答题卡、答题纸等）、草稿纸等考试用纸带出考场；用规定以外的笔或纸答题，或在试卷规定以外的地方书写姓名、考号，或以其他方式在答卷上标记信息；其他违反考场规则但尚未构成作弊的行为。",
        body_style
    ))

    story.append(Paragraph("第十五条 申诉程序", section_style))
    story.append(Paragraph(
        "学生对处分决定有异议的，在接到学校处分决定书之日起10个工作日内，可以向学校学生申诉处理委员会提出书面申诉。学生申诉处理委员会对学生提出的申诉进行复查，并在接到书面申诉之日起15个工作日内，作出复查结论并告知申诉人。",
        body_style
    ))

    # 生成PDF
    doc.build(story)
    print(f"示例学生手册已创建: {output_path.absolute()}")

    return output_path


if __name__ == "__main__":
    create_sample_handbook()
