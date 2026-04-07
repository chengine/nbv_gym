window.HELP_IMPROVE_VIDEOJS = false;

class BeforeAfter {
    constructor(entryObject) {
        const container = document.querySelector(entryObject.id);
        const before = container.querySelector('.bal-before');
        const handle = container.querySelector('.bal-handle');
        const beforeInset = container.querySelector('.bal-before-inset');

        beforeInset.setAttribute('style', 'width: ' + container.offsetWidth + 'px;');
        window.addEventListener('resize', function() {
            beforeInset.setAttribute('style', 'width: ' + container.offsetWidth + 'px;');
        });

        before.setAttribute('style', 'width: 50%;');
        handle.setAttribute('style', 'left: 50%;');

        container.addEventListener('touchstart', function() {
            container.addEventListener('touchmove', function(e2) {
                var containerWidth = container.offsetWidth;
                var currentPoint = e2.changedTouches[0].clientX;
                var startOfDiv = container.getBoundingClientRect().left;
                var modifiedCurrentPoint = currentPoint - startOfDiv;
                if (modifiedCurrentPoint > 10 && modifiedCurrentPoint < containerWidth - 10) {
                    var newWidth = modifiedCurrentPoint * 100 / containerWidth;
                    before.setAttribute('style', 'width:' + newWidth + '%;');
                    handle.setAttribute('style', 'left:' + newWidth + '%;');
                }
            });
        });

        container.addEventListener('mousemove', function(e) {
            var containerWidth = container.offsetWidth;
            var rect = container.getBoundingClientRect();
            var offsetX = e.clientX - rect.left;
            if (offsetX > 10 && offsetX < containerWidth - 10) {
                var newWidth = offsetX * 100 / containerWidth;
                before.setAttribute('style', 'width:' + newWidth + '%;');
                handle.setAttribute('style', 'left:' + newWidth + '%;');
            }
        });
    }
}

$(document).ready(function() {
    bulmaSlider.attach();

    // Initialize comparison sliders
    $('.bal-container').each(function(i) {
        new BeforeAfter({ id: '#' + this.id });
    });

    // Scene selector
    var selector = document.getElementById('scene-selector');
    if (selector) {
        selector.addEventListener('change', function() {
            var scene = this.value;
            $('.bal-container').each(function() {
                var method = $(this).data('method');
                $(this).find('.img-gt').attr('src', 'assets/' + scene + '/ground_truth.png');
                $(this).find('.img-method').attr('src', 'assets/' + scene + '/' + method + '.png');
            });
        });
    }
})

function copyBibTeX() {
    var bibtexElement = document.getElementById('bibtex-code');
    var button = document.querySelector('.copy-bibtex-btn');
    var copyText = button.querySelector('.copy-text');
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            button.classList.add('copied');
            copyText.textContent = 'Copied!';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}
