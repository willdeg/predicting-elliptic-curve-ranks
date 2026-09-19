"""Run the original elliptic-curve classification experiments on the bundled CSV.

Ported from Vincent's Colab export. Model settings and historical subset
slices are preserved; splits use seed 42 and figures are saved to outputs/.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def save_figure():
    """Save each experiment plot separately without requiring a GUI."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"figure_{save_figure.count:02d}.png", dpi=160)
    save_figure.count += 1
    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_figure.count = 1
    df2 = pd.read_csv(PROJECT_ROOT / "elliptic_curve_data.csv")
    df2.head()

    y=df2["4"]
    print(y)

    a_p=[str(i) for i in range(6, 206)]
    X=df2[a_p]
    X.head()

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=.2, random_state=42)
    print(X_train)
    len(X_train)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train,Y_train)

    Y_pred=logreg.predict(X_test)

    accuracy=accuracy_score(Y_pred,Y_test)
    print(accuracy)
    print("\nConfusion matrix:\n", confusion_matrix(Y_test, Y_pred))

    """# graph the data"""

    rankzerocurves=df2[df2["4"]==0]

    rankzerocurves.head()

    print(len(rankzerocurves))

    rankonecurves=df2[df2["4"]==1]

    rankonecurves.head()

    ranktwocurves=df2[df2["4"]==2]

    ranktwocurves.head()

    print(len(ranktwocurves))

    averagea2r0=rankzerocurves["6"].mean()
    print(averagea2r0)

    rank0traces=[]
    for i in range(6,206):
      rank0traces.append(rankzerocurves[str(i)].mean())

    print(rank0traces)

    rank1traces=[]
    for i in range(6,206):
      rank1traces.append(rankonecurves[str(i)].mean())

    print(rank1traces)

    rank2traces=[]
    for i in range(6,206):
      rank2traces.append(ranktwocurves[str(i)].mean())

    import matplotlib.pyplot as plt
    x = list(range(1, 201))
    plt.scatter(x, rank0traces, color="red", s=30, alpha=0.7, label="rank 0" )
    plt.scatter(x, rank1traces, color="blue", s=30, alpha=0.7, label="rank 1" )
    plt.scatter(x, rank2traces, color="green", s=30, alpha=0.7, label="rank 2")
    plt.xlabel("Prime index (first 200 primes)")
    plt.ylabel("traces")
    plt.title("Average Frobenius traces for ranks 0, 1, and 2")
    plt.legend()
    save_figure()

    conductor=df2["0"]
    print(conductor)

    lowest_1k  = conductor.nsmallest(1000)
    highest_1k = conductor.nlargest(1000)

    sortrankzerocurves=rankzerocurves.sort_values(by="0")

    sortrankzerocurves.head(15)

    lowest_1kr0=sortrankzerocurves.head(1000)
    largest_1kr0=sortrankzerocurves.tail(1000)
    print(len(lowest_1kr0))
    print(len(sortrankzerocurves))

    lowestrank0traces=[]
    for i in range(6,206):
      lowestrank0traces.append(lowest_1kr0[str(i)].mean())

    largestrank0traces=[]
    for i in range(6,206):
      largestrank0traces.append(largest_1kr0[str(i)].mean())

    import matplotlib.pyplot as plt
    x = list(range(1, 201))
    plt.scatter(x, lowestrank0traces, color="red", s=30, alpha=0.7, label="low conductor" )
    plt.scatter(x, largestrank0traces, color="blue", s=30, alpha=0.7, label="large conductor" )
    plt.xlabel("Prime index (first 200 primes)")
    plt.ylabel("traces")
    plt.title("Low vs High conductor rank 0 curves")
    plt.legend()
    save_figure()

    sortrankonecurves=rankonecurves.sort_values(by="0")

    lowest_1kr1=sortrankonecurves.head(1000)
    largest_1kr1=sortrankonecurves.tail(1000)
    lowestrank1traces=[]
    for i in range(6,206):
      lowestrank1traces.append(lowest_1kr1[str(i)].mean())
    largestrank1traces=[]
    for i in range(6,206):
      largestrank1traces.append(largest_1kr1[str(i)].mean())

    import matplotlib.pyplot as plt
    x = list(range(1, 201))
    plt.scatter(x, lowestrank1traces, color="red", s=30, alpha=0.7, label="low conductor" )
    plt.scatter(x, largestrank1traces, color="blue", s=30, alpha=0.7, label="large conductor" )
    plt.xlabel("Prime index (first 200 primes)")
    plt.ylabel("traces")
    plt.title("Low vs High conductor rank 1 curves")
    plt.legend()
    save_figure()

    import matplotlib.pyplot as plt
    x = list(range(1, 201))
    plt.scatter(x, lowestrank1traces, color="red", s=30, alpha=0.7, label="r1 low conductor" )
    plt.scatter(x, lowestrank0traces, color="blue", s=30, alpha=0.7, label="r0 low conductor" )
    plt.xlabel("Prime index (first 200 primes)")
    plt.ylabel("traces")
    plt.title("Rank 0 vs Rank 1 low conductor curves")
    plt.legend()
    save_figure()

    import matplotlib.pyplot as plt
    x = list(range(1, 201))
    plt.scatter(x, largestrank1traces, color="red", s=30, alpha=0.7, label="r1 large conductor" )
    plt.scatter(x, largestrank0traces, color="blue", s=30, alpha=0.7, label="r0 large conductor" )
    plt.xlabel("Prime index (first 200 primes)")
    plt.ylabel("traces")
    plt.title("Rank 0 vs Rank 1 large conductor curves")
    plt.legend()
    save_figure()

    rank0conductor=rankzerocurves["0"]
    rank1conductor=rankonecurves["0"]
    largest_1kr0.tail(5)
    largest_1kr1.tail(5)
    largest_1kr0.head(5)

    merged_largest = pd.concat([largest_1kr0, largest_1kr1], ignore_index=True)
    merged_largest.head()
    y1=merged_largest["4"]
    a_p=[str(i) for i in range(6, 206)]
    X1=merged_largest[a_p]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    X_train, X_test, Y_train, Y_test=train_test_split(X1,y1,test_size=.2, random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train,Y_train)
    Y_pred=logreg.predict(X_test)
    accuracy1=accuracy_score(Y_pred,Y_test)
    print(accuracy1)

    merged_lowest = pd.concat([lowest_1kr0, lowest_1kr1], ignore_index=True)
    merged_lowest.head()
    y=merged_lowest["4"]
    a_p=[str(i) for i in range(6, 206)]
    X=merged_lowest[a_p]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=.2, random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train,Y_train)
    Y_pred=logreg.predict(X_test)
    accuracy2=accuracy_score(Y_pred,Y_test)
    print(accuracy2)

    original=pd.concat([lowest_1kr0, lowest_1kr1, largest_1kr0, largest_1kr1], ignore_index=True)
    y=original["4"]
    a_p=[str(i) for i in range(6, 206)]
    X=original[a_p]
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=.2, random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train,Y_train)
    Y_pred=logreg.predict(X_test)
    accuracy3=accuracy_score(Y_pred,Y_test)
    print(accuracy3)

    #make it merge 1000 rank 0 (200-1200) and the same for rank 1. (You are keeping 1000 conductor with mixed rank which is wrong)
    r0_200_1200 = sortrankzerocurves.iloc[200:1201, :].copy()
    r1_200_1200 = sortrankonecurves.iloc[200:1201, :].copy()
    merged_200_1200 = pd.concat([r0_200_1200, r1_200_1200], ignore_index=True)
    y=merged_200_1200["4"]
    a_p=[str(i) for i in range(6, 206)]
    X=merged_200_1200[a_p]
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=.2, random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train,Y_train)
    Y_pred=logreg.predict(X_test)
    accuracy4=accuracy_score(Y_pred,Y_test)
    print(accuracy4)

    r0_400_1400 = sortrankzerocurves.iloc[400:1401, :].copy()
    r1_400_1400 = sortrankonecurves.iloc[400:1401, :].copy()
    merged_400_1400 = pd.concat([r0_400_1400, r1_400_1400], ignore_index=True)
    y=merged_400_1400["4"]
    a_p=[str(i) for i in range(6, 206)]
    X=merged_400_1400[a_p]
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=.2, random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train,Y_train)
    Y_pred=logreg.predict(X_test)
    accuracy5=accuracy_score(Y_pred,Y_test)
    print(accuracy5)

    r0_600_1600 = sortrankzerocurves.iloc[600:1601, :].copy()
    r1_600_1600 = sortrankonecurves.iloc[600:1601, :].copy()
    merged_600_1600 = pd.concat([r0_600_1600, r1_600_1600], ignore_index=True)
    y=merged_600_1600["4"]
    a_p=[str(i) for i in range(6, 206)]
    X=merged_600_1600[a_p]
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=.2, random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train,Y_train)
    Y_pred=logreg.predict(X_test)
    accuracy6=accuracy_score(Y_pred,Y_test)
    print(accuracy6)

    #do the 800 one
    r0_800_1800 = sortrankzerocurves.iloc[800:1801, :].copy()
    r1_800_1800 = sortrankonecurves.iloc[800:1801, :].copy()
    merged_800_1800 = pd.concat([r0_800_1800, r1_800_1800], ignore_index=True)
    y=merged_800_1800["4"]
    a_p=[str(i) for i in range(6, 206)]
    X=merged_800_1800[a_p]
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=.2, random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train,Y_train)
    Y_pred=logreg.predict(X_test)
    accuracy7=accuracy_score(Y_pred,Y_test)
    print(accuracy7)

    # Preserve the original final window: indices 1200 through 2200.
    r0_1000_2000 = sortrankzerocurves.iloc[1200:2201, :].copy()
    r1_1000_2000 = sortrankonecurves.iloc[1200:2201, :].copy()
    merged_1000_2000 = pd.concat([r0_1000_2000, r1_1000_2000], ignore_index=True)
    y=merged_1000_2000["4"]
    a_p=[str(i) for i in range(6, 206)]
    X=merged_1000_2000[a_p]
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=.2, random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train,Y_train)
    Y_pred=logreg.predict(X_test)
    accuracy8=accuracy_score(Y_pred,Y_test)
    print(accuracy8)

    avg_low  = merged_lowest["0"].mean()
    avg_high = merged_largest["0"].mean()
    avg_middle1=merged_200_1200["0"].mean()
    avg_middle2=merged_400_1400["0"].mean()
    avg_middle3=merged_600_1600["0"].mean()
    avg_middle4=merged_800_1800["0"].mean()
    avg_middle5=merged_1000_2000["0"].mean()
    X=[avg_low,avg_middle1,avg_middle2,avg_middle3,avg_middle4, avg_middle5,avg_high]
    Y=[accuracy2,accuracy4,accuracy5,accuracy6,accuracy7,accuracy8,accuracy1]

    plt.figure(figsize=(7,5))
    plt.plot(X, Y, 'o-', color='blue', markersize=8, linewidth=2)
    plt.title("Accuracy vs Average Conductor", fontsize=14)
    plt.xlabel("Average Conductor of Training Set", fontsize=12)
    plt.ylabel("Model Accuracy", fontsize=12)
    for i in range(len(X)):
        plt.text(X[i], Y[i]+0.005, f"({X[i]:.0f}, {Y[i]:.2f})", ha='center')

    plt.grid(True)
    save_figure()

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    # Compute confusion matrix (raw counts)
    cm = confusion_matrix(Y_test, Y_pred, labels=[0,1])

    # Display raw confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Rank 0', 'Rank 1'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix for Logistic Regression (Rank 0 vs Rank 1)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_rank01.png", dpi=300)
    save_figure()

    # Compute and display normalized confusion matrix (percentages)
    cm_norm = confusion_matrix(Y_test, Y_pred, labels=[0,1], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['Rank 0', 'Rank 1'])
    disp.plot(cmap='Blues', values_format='.2f')
    plt.title("Normalized Confusion Matrix for Logistic Regression (Rank 0 vs Rank 1)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_rank01_normalized.png", dpi=300)
    save_figure()


if __name__ == "__main__":
    main()
