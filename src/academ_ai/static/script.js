let tooltipIsDisplayed = false;
let tooltipTimeout = null;

const highlightColor = "#b9ff9e";
const tooltipBackgroundColor = "#6cfa34";
const PROTOCOL = "http";
const API_HOST = "{{ API_HOST }}";
const API_PORT = "{{ API_PORT }}";

splitAtIndices = function(str, indices) {
    indices = indices.sort((a, b) => a - b);
    if (indices[0] !== 0) {
        indices.unshift(0);
    }
    if (indices[indices.length - 1] !== str.length) {
        indices.push(str.length);
    }
    
    const parts = {};
    for (let i = 0; i < indices.length - 1; i++) {
        parts[indices[i]] = str.substring(indices[i], indices[i + 1]);
    }
    return parts;
}

makeQuotePrettier = function(quote, maxLen) {
    [...quote.matchAll(titleRegex)].forEach((match) => {
        quote = quote.replace(match[0], `<b class="paper-subtitle">${match[0]}.</b> `);
    });
    const words = quote.split(" ");
    if (words.length <= maxLen) {
        return quote;
    }
    var collapsed = words.slice(0, maxLen / 2).join(" ") + " (...) " + words.slice(-maxLen / 2).join(" ");
    collapsed = formatText(collapsed);
    return collapsed;
}

formatText = function(text) {
    text = text.replace(/\n+/g, "\n");
    text = text.replace(/\n/g, "<br><br>");
    text = text.split("Graphical Abstract")[0];
    text = text.split("Graphical abstract")[0];
    return text;
}

window.queryText = async function(query_text, alpha, number) {
    const response = await fetch(`${PROTOCOL}://${API_HOST}:${API_PORT}/query`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({"query_text": query_text, "limit": parseInt(number), "query_kwargs": {"alpha": parseFloat(alpha)}}),
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

window.makePaperCard = function(result) {
    const paper = document.createElement("div");
    paper.className = 'paper-card';
    const abstractText = result.abstract || 'No abstract available';
    // This process assumes the spans are sorted (they come sorted from the API)
    var flatSpans = Array();
    if (result.spans.length < 2) {
        for (const [idx, span] of result.spans.entries()) {
            flatSpans.push({"span": span, "idxs": [idx]})
        }
    } else {
        for (const [idx1, span1] of result.spans.entries()) {
            for (const idx2 of [...Array(result.spans.length).keys()].slice(idx1+1)) {
                const span2 = result.spans[idx2];
                if ((span2[0] > span1[0]) && (span2[0] < span1[1])) {
                    const spanA = [span1[0], span2[0]];
                    const spanB = [span2[0], Math.min(span1[1], span2[1])];
                    var spanC = Array();
                    if (span1[1] > span2[1]) {
                        spanC = [span2[1], span1[1]];
                    } else {
                        spanC = [span1[1], span2[1]];
                    }
                    flatSpans.push({"span": spanA, "idxs": [idx1]});
                    flatSpans.push({"span": spanB, "idxs": [idx1, idx2]});
                    flatSpans.push({"span": spanC, "idxs": [idx2]});
                } else {
                    flatSpans.push({"span": span1, "idxs": [idx1]});
                    flatSpans.push({"span": span2, "idxs": [idx2]});
                }
            }
            
        }
    }
    var breakPoints = Array();
    for (const span of flatSpans) {
        if (!breakPoints.includes(span.span[0])) {
            breakPoints.push(span.span[0])
        }
        if (!breakPoints.includes(span.span[1])) {
            breakPoints.push(span.span[1])
        }
    }
    breakPoints.sort((a, b) => a - b);
    // Remove breakpoints whose distance to the next is smaller than one
    breakPoints = breakPoints.filter((breakPoint, idx) => {
        if (idx == breakPoints.length - 1) {
            return true;
        }
        return breakPoints[idx + 1] - breakPoint > 1;
    });
    const splitAbstract = splitAtIndices(abstractText, breakPoints)
    const abstractClasses = {}
    let finalAbstract = ""
    for (var [abstractIdx, abstractPart] of Object.entries(splitAbstract)) {
        const currSpans = flatSpans.filter((span) => span.span[0] == abstractIdx);
        var spanBefore = "";
        var spanAfter = "";
        if (currSpans.length > 0) {
            const ids = currSpans[0].idxs.map((idx) => result.paper_id + "_" + idx);
            spanBefore += `<span class="paper-highlight ${ids.join(" ")}">`;
            spanAfter += `</span>`;
            abstractClasses[abstractIdx] = ids
        }
        [...abstractPart.matchAll(titleRegex)].forEach((match) => {
            abstractPart = abstractPart.replace(match[0], `<b>${match[0]}.</b> `);
        });
        finalAbstract += spanBefore + abstractPart + spanAfter;
    }
    finalAbstract = formatText(finalAbstract);
    const maxScore = Math.max(...result.scores);
    const title = `<h2 class="paper-title">${result.title || 'No title available'}</h2>`;
    const abstract = `<div class="paper-abstract">${finalAbstract}</div>`;
    const authors = `<p class="paper-authors">${result.authors.join(", ")}</p>`;
    paper.innerHTML = `
        <div class="paper-content">
            ${title}
            ${authors}
            ${abstract}
            <div class="paper-footer">
                <div class="badge">
                    <b>Maximum score:</b> ${maxScore.toFixed(2)}
                </div>
                <a href="https://doi.org/${result.doi}" target="_blank" rel="noopener noreferrer">
                    <div class="badge badge-clickable float-right">
                        <span class="paper-link">
                            View Paper
                            <i class="fas fa-external-link-alt fa-2xs"></i>
                        </span>
                    </div>
                </a>
            </div>
        </div>
    `;
    return paper;
}

window.highlightText = function(event, quotes) {
    var span = event.target;
    if (span.tagName == "B") {
        span = span.parentElement;
    }
    span.style.backgroundColor = highlightColor;
    span.classList.forEach((cl) => {
        if (cl != "paper-highlight") {
            Object.values(document.getElementsByClassName(cl)).forEach((el) => {
                el.style.backgroundColor = highlightColor;
            });
        }
    })
    const tooltip = document.getElementById("tooltip");
    let vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
    let maxQuoteLength = 60;
    if (vw < 600) {
        maxQuoteLength = 30;
    }
    tooltip.innerHTML = quotes.map((quote) => {
        reranking_score_str = "";
        if (quote.explain_score.reranking_score) {
            reranking_score_str = `<p class="badge"><b>Reranking score:</b> ${quote.explain_score.reranking_score.toFixed(2)}</p>`;
        }
        return `
        <div class="paper-quote-container">
            <p class="paper-quote">"${makeQuotePrettier(quote.quote, maxQuoteLength)}"</p>
            <div class="paper-footer paper-scores-tooltip">
                <p class="badge"><b>Overall score:</b> ${quote.score.toFixed(2)}</p>
                <p class="badge">
                    <b>KW score:</b> ${(quote.explain_score.keyword || 0).toFixed(2)} | 
                    <b>Semantic score:</b> ${(quote.explain_score.semantic || 0).toFixed(2)}
                </p>
                ${reranking_score_str}
            </div>
        </div>
    `
    }).join("");

    tooltip.style.display = "block";

    // Position the tooltip above or below the mouse pointer
    const rect = span.getBoundingClientRect();
    const tooltipHeight = Math.max(tooltip.offsetHeight, 150);
    const viewportHeight = window.innerHeight;
    const N = 30;
    
    // Default to positioning below the cursor
    let topPosition = rect.top + rect.height + N;

    // If there's not enough space below, position above the cursor
    if ((topPosition + tooltipHeight) > (viewportHeight - N)) {
        topPosition = rect.top - (tooltipHeight + N);
    }
    
    tooltip.style.top = topPosition + 'px';
    tooltip.style.animationDelay = "0.3s";
    tooltip.style.animationName = "from-right";

    tooltipIsDisplayed = false;
    tooltipTimeout = setTimeout(() => {
        tooltipIsDisplayed = true;
    }, 600);
}

window.unHighlightText = function(event) {
    var span = event.target;
    if (span.tagName == "B") {
        span = span.parentElement;
    }
    span.style.backgroundColor = tooltipBackgroundColor;
    span.classList.forEach((cl) => {
        if (cl != "paper-highlight") {
            Object.values(document.getElementsByClassName(cl)).forEach((el) => {
                el.style.backgroundColor = tooltipBackgroundColor;
            });
        }
    })
    const tooltip = document.getElementById("tooltip");
    if (tooltipIsDisplayed) {
        tooltip.style.animationDelay = "0s";
        tooltip.style.animationName = "to-left";
    } else {
        if (tooltipTimeout) {
            clearTimeout(tooltipTimeout);
        }
        tooltip.style.animationName = null;
        tooltip.style.visibility = "hidden";
    }
}
