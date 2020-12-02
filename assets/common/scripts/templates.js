define(_require => {
  function registerTemplateById(id) {
    // SRC - https://developer.mozilla.org/en-US/docs/Web/Web_Components/Using_templates_and_slots

    customElements.define(id,
      class extends HTMLElement {
        constructor() {
          super();

          let template = document.getElementById(id);
          let templateContent = template.content;

          const shadowRoot = this.attachShadow({
              mode: 'open'
            })
            .appendChild(templateContent.cloneNode(true));
        }
      }
    );
  }

  function registerAllTemplates(parent = document) {
    const templates = Array.from(parent.getElementsByTagName('template'));
    templates.forEach(t => registerTemplateById(t.id))
  }

  return {
    registerTemplateById,
    registerAllTemplates,
  };
});
