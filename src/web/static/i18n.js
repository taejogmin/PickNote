/* 한국어 / English 전환.
 *
 * 기본 언어는 서버가 <html lang="…">에 넣어 준다 (0_Setup.cmd에서 고른 것).
 * 화면에서 버튼으로 바꾸면 이 브라우저가 기억한다.
 *
 *   L("한국어", "English")         — 스크립트가 만드는 문장
 *   <span data-en="English">한국어</span>   — 화면에 박힌 문장 (data-en-html이면 innerHTML)
 *   data-en-placeholder / data-en-title     — 속성
 *   champName(c)                   — champions.json 항목의 이름 (c.name 한글 · c.en 영어)
 */
(function () {
  const KEY = "picknote.lang";
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) { /* 사생활 보호 창 */ }
  const LANG = saved === "ko" || saved === "en" ? saved : (document.documentElement.lang === "en" ? "en" : "ko");
  document.documentElement.lang = LANG;

  window.LANG = LANG;
  window.L = (ko, en) => (LANG === "en" ? en : ko);
  window.champName = c => (c ? (LANG === "en" ? (c.en || c.name) : c.name) : "");

  window.applyI18n = function (root) {
    if (LANG !== "en") return;
    (root || document).querySelectorAll("[data-en],[data-en-html],[data-en-placeholder],[data-en-title]").forEach(el => {
      if (el.dataset.en !== undefined) el.textContent = el.dataset.en;
      if (el.dataset.enHtml !== undefined) el.innerHTML = el.dataset.enHtml;
      if (el.dataset.enPlaceholder !== undefined) el.placeholder = el.dataset.enPlaceholder;
      if (el.dataset.enTitle !== undefined) el.title = el.dataset.enTitle;
    });
  };

  // Riot 정책: 이 문구를 "사용자 눈에 잘 띄는 곳"에 둬야 한다. 문구가 정해져 있어 언어와 상관없이 원문 그대로 둔다.
  const LEGAL = "PickNote isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games " +
    "or anyone officially involved in producing or managing Riot Games properties. " +
    "Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.";

  function addLegal() {
    const style = document.createElement("style");
    style.textContent = "#riot-legal{max-width:960px;margin:24px auto 0;padding:14px 120px 18px 16px;" +
      "font-size:11px;line-height:1.5;color:var(--faint,#7b8494);opacity:.85}";
    document.head.appendChild(style);
    const foot = document.createElement("footer");
    foot.id = "riot-legal";
    foot.textContent = LEGAL;
    document.body.appendChild(foot);
  }

  function addToggle() {
    addLegal();
    const btn = document.createElement("button");
    btn.id = "lang-toggle";
    btn.type = "button";
    btn.textContent = LANG === "en" ? "한국어" : "English";
    btn.title = LANG === "en" ? "한국어로 보기" : "Switch to English";
    btn.addEventListener("click", () => {
      try { localStorage.setItem(KEY, LANG === "en" ? "ko" : "en"); } catch (e) { /* 저장 못 해도 이번만 바꾼다 */ }
      location.reload();
    });
    document.body.appendChild(btn);
    applyI18n(document);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", addToggle);
  else addToggle();
})();
