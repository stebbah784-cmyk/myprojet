with tab2:

    st.subheader("☁️ WordCloud Analysis")

    option = st.selectbox(
        "Select sentiment:",
        ["All", "Positive", "Negative", "Neutral"]
    )

    if option != "All":
        data_words = df_filtered[
            df_filtered["AI_Sentiment"] == option
        ]
    else:
        data_words = df_filtered

    # =====================
    # SAFE TEXT CREATION
    # =====================
    text = " ".join(
        data_words["Clean_Tweet"].dropna().astype(str)
    )

    if text.strip() == "" or len(text.split()) < 2:
        st.warning("⚠️ No enough words to generate WordCloud for this selection.")
    else:
        wc = WordCloud(
            width=1200,
            height=500,
            background_color="white",
            colormap="viridis"
        ).generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # =====================
        # TOP WORDS SAFE
        # =====================
        st.markdown("### 🔥 Top Words")

        words = text.split()

        if len(words) == 0:
            st.warning("No words found.")
        else:
            common = Counter(words).most_common(15)

            words_df = pd.DataFrame(common, columns=["Word", "Count"])

            fig2 = px.bar(
                words_df,
                x="Word",
                y="Count",
                title="Most Frequent Words"
            )

            st.plotly_chart(fig2, use_container_width=True)
