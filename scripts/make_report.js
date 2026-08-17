const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  BorderStyle, VerticalAlign
} = require("docx");
const fs = require("fs");

const NAVY = "1F3864";
const GREEN = "2E8B57";
const GRAY = "595959";

function h(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, color: NAVY, size: 28 })],
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, size: 22, ...opts })],
  });
}

function bullet(text) {
  return new Paragraph({
    bullet: { level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, size: 22 })],
  });
}

function statBox(label, value, color) {
  return new TableCell({
    width: { size: 33, type: WidthType.PERCENTAGE },
    shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 150, bottom: 150, left: 100, right: 100 },
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: value, bold: true, size: 34, color })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: label, size: 18, color: GRAY })],
      }),
    ],
  });
}

const doc = new Document({
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 }, // A4
          margin: { top: 900, bottom: 900, left: 1100, right: 1100 },
        },
      },
      children: [
        // TITLE
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 60 },
          children: [
            new TextRun({ text: "Mənzil Qiymətinin Avtomatik Proqnozlaşdırılması",
              bold: true, size: 34, color: NAVY }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 260 },
          children: [
            new TextRun({ text: "Bina.az elan datası əsasında hazırlanmış qiymət təxmini modeli — yekun xülasə",
              italics: true, size: 20, color: GRAY }),
          ],
        }),

        // PROBLEM
        h("Məsələ nədir?"),
        p("Bina.az-da hər gün minlərlə mənzil elanı yerləşdirilir. Nə satıcı, nə də alıcı üçün " +
          "bir mənzilin \"ədalətli\" bazar qiymətinin nə olduğunu dəqiq bilmək asan deyil — bu, adətən " +
          "oxşar elanları əl ilə gəzib müqayisə etməklə edilir. Bu həm vaxt aparır, həm də subyektivdir."),
        p("Biz mənzilin əsas xüsusiyyətlərinə (sahə, otaq sayı, rayon, mərtəbə, təmir vəziyyəti və s.) " +
          "əsasən onun təxmini bazar qiymətini saniyələr içində hesablayan bir alət hazırladıq."),

        // WHO BENEFITS
        h("Bundan kim və necə faydalanacaq?"),
        bullet("Elan yerləşdirən istifadəçilər — real bazara uyğun qiymət tövsiyəsi alacaq."),
        bullet("Alıcılar — gördükləri elanın bazar ortalamasına nisbətən baha, ya ucuz olduğunu dərhal görəcək."),
        bullet("Bina.az — platformaya \"Təxmini qiymət\" funksiyası əlavə edərək istifadəçi təcrübəsini gücləndirəcək."),

        // ACCURACY - big stat boxes
        h("Model nə qədər dəqiqdir?"),
        p("Model, əvvəllər görmədiyi 11 000+ real elan üzərində sınaqdan keçirildi. Nəticələr:"),
        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          borders: {
            top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
            left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
            insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
          },
          rows: [
            new TableRow({
              children: [
                statBox("orta xəta payı", "9.4%", GREEN),
                statBox("uyğunluq göstəricisi", "93%", GREEN),
                statBox("sınaqdan keçirilmiş elan", "11 090", NAVY),
              ],
            }),
          ],
        }),
        new Paragraph({ spacing: { before: 180, after: 120 }, children: [
          new TextRun({ text: "Sadə dillə desək: model orta hesabla real qiymətdən cəmi ~9% kənarlaşma ilə " +
            "proqnoz verir. Məsələn, 150 000 AZN dəyərində mənzil üçün model adətən 136 000–164 000 AZN " +
            "aralığında (yəni faktiki qiymətə çox yaxın) təxmin göstərir.", size: 22 }),
        ]}),

        new Paragraph({
          children: [new ImageRun({
            type: "png",
            data: fs.readFileSync("/home/claude/proj/report_figs/r1_accuracy.png"),
            transformation: { width: 380, height: 285 },
          })],
          alignment: AlignmentType.CENTER,
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "Nöqtələr qırmızı xəttə nə qədər yaxındırsa, proqnoz bir o qədər dəqiqdir.",
            italics: true, size: 18, color: GRAY })],
        }),

        // HOW WE GOT THERE
        h("Necə bu nəticəyə gəldik?"),
        p("Dörd fərqli proqnozlaşdırma üsulu eyni şərtlər altında sınaqdan keçirildi və ən yaxşı nəticə " +
          "verən üsul seçilib əlavə tənzimləndi:"),
        new Paragraph({
          children: [new ImageRun({
            type: "png",
            data: fs.readFileSync("/home/claude/proj/report_figs/r2_model_compare.png"),
            transformation: { width: 420, height: 260 },
          })],
          alignment: AlignmentType.CENTER,
        }),
        new Paragraph({ spacing: { before: 120, after: 200 }, children: [
          new TextRun({ text: "Yaşıl zolaqla göstərilən üsul (istifadə edilən yekun model) sürət və dəqiqlik " +
            "arasında ən yaxşı balansı təmin etdiyi üçün seçildi.", size: 22 }),
        ]}),

        // WHAT DRIVES PRICE
        h("Qiyməti ən çox nə müəyyən edir?"),
        new Paragraph({
          children: [new ImageRun({
            type: "png",
            data: fs.readFileSync("/home/claude/proj/report_figs/r3_importance.png"),
            transformation: { width: 420, height: 260 },
          })],
          alignment: AlignmentType.CENTER,
        }),
        p("Gözlənildiyi kimi, sahə (m²) ən güclü amildir. Ondan sonra yerləşmə (hansı hissədə " +
          "olması) gəlir — bu, modelin \"mərkəzə yaxın rayonlar daha bahadır\" kimi real bazar " +
          "qanunauyğunluqlarını düzgün öyrəndiyini göstərir."),

        // LIMITATIONS
        h("Nələrə diqqət etmək lazımdır?"),
        bullet("Model hazırda yalnız Bakı şəhərində, Yeni tikili və Köhnə tikili mənzil elanları üçün öyrədilib — " +
          "həyət evi, torpaq və kommersiya obyektləri əhatə olunmur."),
        bullet("Model tarixi elan qiymətlərinə əsaslanır, real satış qiymətlərinə deyil — buna görə bazar " +
          "trendinin sürətlə dəyişdiyi dövrlərdə (məsələn, kəskin inflyasiya) yenidən öyrədilməlidir."),
        bullet("Nadir/unikal mənzillər (məsələn, çox baha lüks mənzillər) üçün proqnoz dəqiqliyi bir qədər aşağı ola bilər."),

        // NEXT STEPS
        h("Növbəti addımlar"),
        bullet("Modeli test mühitində Bina.az platformasına inteqrasiya etmək (elan yerləşdirərkən \"tövsiyə " +
          "olunan qiymət\" göstərmək)."),
        bullet("Digər şəhərləri və əmlak növlərini (ev, torpaq, ofis) əhatə edəcək şəkildə genişləndirmək."),
        bullet("Modeli mütəmadi olaraq (məs. rüblük) yeni data ilə yeniləmək ki, bazar dəyişikliklərini əks etdirsin."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/home/claude/proj/business_report.docx", buf);
  console.log("Report written.");
});
